use anyhow::Context;
use cgmath::{Deg, Euler, Matrix3, Point3, Quaternion, Rad, SquareMatrix, Vector2};
use clap::Parser;
use std::io::{BufRead, BufReader, Write};
use std::{fs::File, path::PathBuf, time::Duration};
use web_splats::{
    GaussianRenderer, PerspectiveCamera, PerspectiveProjection, PointCloud, SplattingArgs,
    WGPUContext, io::GenericGaussianPointCloud,
};

#[derive(Debug, Parser)]
#[command(author, version)]
#[command(
    about = "Persistent headless renderer. Reads 6DoF poses from stdin and writes raw RGBA frames to stdout",
    long_about = None
)]
struct Opt {
    /// Input point cloud (.ply or .npz with feature npz enabled)
    input: PathBuf,

    /// Output width in pixels
    #[arg(long)]
    width: u32,

    /// Output height in pixels
    #[arg(long)]
    height: u32,

    /// Horizontal focal length in pixels
    #[arg(long)]
    fx: f32,

    /// Vertical focal length in pixels
    #[arg(long)]
    fy: f32,

    /// Render background red component in [0, 1]
    #[arg(long, default_value_t = 0.0)]
    bg_r: f64,

    /// Render background green component in [0, 1]
    #[arg(long, default_value_t = 0.0)]
    bg_g: f64,

    /// Render background blue component in [0, 1]
    #[arg(long, default_value_t = 0.0)]
    bg_b: f64,
}

#[derive(Debug, Clone, Copy)]
struct Pose6D {
    x: f32,
    y: f32,
    z: f32,
    roll_deg: f32,
    pitch_deg: f32,
    yaw_deg: f32,
}

impl Pose6D {
    fn parse(values: &[f32]) -> anyhow::Result<Self> {
        if values.len() != 6 {
            anyhow::bail!("expected 6 values for Euler pose, received {}", values.len());
        }

        Ok(Self {
            x: values[0],
            y: values[1],
            z: values[2],
            roll_deg: values[3],
            pitch_deg: values[4],
            yaw_deg: values[5],
        })
    }

    fn to_camera(self, width: u32, height: u32, fx: f32, fy: f32) -> PerspectiveCamera {
        let fovx = focal_to_fov(fx, width as f32);
        let fovy = focal_to_fov(fy, height as f32);
        let projection =
            PerspectiveProjection::new(Vector2::new(width, height), Vector2::new(fovx, fovy), 0.01, 1000.0);
        let rotation = Quaternion::from(Euler {
            x: Deg(self.pitch_deg),
            y: Deg(self.yaw_deg),
            z: Deg(self.roll_deg),
        });

        PerspectiveCamera::new(
            Point3::new(self.x, self.y, self.z),
            rotation,
            projection,
        )
    }
}

fn parse_camera(line: &str, width: u32, height: u32, fx: f32, fy: f32) -> anyhow::Result<PerspectiveCamera> {
    let values: Vec<f32> = line
        .split_whitespace()
        .map(str::parse::<f32>)
        .collect::<Result<Vec<_>, _>>()
        .context("failed to parse pose values")?;

    match values.len() {
        6 => Ok(Pose6D::parse(values.as_slice())?.to_camera(width, height, fx, fy)),
        12 => {
            let projection = PerspectiveProjection::new(
                Vector2::new(width, height),
                Vector2::new(focal_to_fov(fx, width as f32), focal_to_fov(fy, height as f32)),
                0.01,
                1000.0,
            );

            let mut rot = Matrix3::from([
                [values[3], values[4], values[5]],
                [values[6], values[7], values[8]],
                [values[9], values[10], values[11]],
            ]);
            if rot.determinant() < 0.0 {
                // Match scene camera conversion behavior for left-handed inputs.
                rot.x[1] = -rot.x[1];
                rot.y[1] = -rot.y[1];
                rot.z[1] = -rot.z[1];
            }

            Ok(PerspectiveCamera {
                position: Point3::new(values[0], values[1], values[2]),
                rotation: rot.into(),
                projection,
            })
        }
        _ => anyhow::bail!(
            "expected either 6 values (x y z roll pitch yaw) or 12 values (x y z + row-major 3x3 rotation), received {}",
            values.len()
        ),
    }
}

struct StreamRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: GaussianRenderer,
    pc: PointCloud,
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    staging_buffer: wgpu::Buffer,
    bytes_per_row: u32,
    width: u32,
    height: u32,
    fx: f32,
    fy: f32,
    background_color: wgpu::Color,
}

impl StreamRenderer {
    async fn new(opt: &Opt) -> anyhow::Result<Self> {
        let ply_file = File::open(&opt.input)
            .with_context(|| format!("failed to open input file {}", opt.input.to_string_lossy()))?;

        let wgpu_context = WGPUContext::new_instance().await;
        let device = wgpu_context.device;
        let queue = wgpu_context.queue;

        let pc_raw = GenericGaussianPointCloud::load(ply_file)
            .with_context(|| format!("failed to read point cloud from {}", opt.input.to_string_lossy()))?;
        let pc = PointCloud::new(&device, pc_raw).context("failed to create GPU point cloud")?;

        let render_format = wgpu::TextureFormat::Rgba8Unorm;
        let renderer =
            GaussianRenderer::new(&device, &queue, render_format, pc.sh_deg(), pc.compressed()).await;

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stream_render target"),
            size: wgpu::Extent3d {
                width: opt.width,
                height: opt.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: render_format,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

        let texel_size = render_format
            .block_copy_size(None)
            .context("failed to query texture format block size")?;
        let align: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1;
        let bytes_per_row = ((texel_size * opt.width) + align) & !align;
        let output_buffer_size = (bytes_per_row * opt.height) as wgpu::BufferAddress;
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stream_render staging buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            renderer,
            pc,
            target,
            target_view,
            staging_buffer,
            bytes_per_row,
            width: opt.width,
            height: opt.height,
            fx: opt.fx,
            fy: opt.fy,
            background_color: wgpu::Color {
                r: opt.bg_r,
                g: opt.bg_g,
                b: opt.bg_b,
                a: 1.0,
            },
        })
    }

    async fn render_camera(&mut self, mut camera: PerspectiveCamera) -> anyhow::Result<Vec<u8>> {
        camera.fit_near_far(self.pc.bbox());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("stream_render encoder"),
            });

        let mut stopwatch = None;
        self.renderer.prepare(
            &mut encoder,
            &self.device,
            &self.queue,
            &self.pc,
            SplattingArgs {
                camera,
                viewport: Vector2::new(self.width, self.height),
                gaussian_scaling: 1.0,
                max_sh_deg: self.pc.sh_deg(),
                mip_splatting: None,
                kernel_size: None,
                clipping_box: None,
                walltime: Duration::from_secs(100),
                scene_center: None,
                scene_extend: None,
                background_color: self.background_color,
            },
            &mut stopwatch,
        );

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stream_render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.target_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.background_color),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.renderer.render(&mut render_pass, &self.pc);
        }

        encoder.copy_texture_to_buffer(
            self.target.as_image_copy(),
            wgpu::TexelCopyBufferInfoBase {
                buffer: &self.staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );

        let submission_idx = self.queue.submit(std::iter::once(encoder.finish()));
        self.download_frame(submission_idx).await
    }

    async fn download_frame(
        &self,
        submission_idx: wgpu::SubmissionIndex,
    ) -> anyhow::Result<Vec<u8>> {
        let slice = self.staging_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_idx),
            timeout: None,
        })?;

        let map_result = rx
            .receive()
            .await
            .ok_or_else(|| anyhow::anyhow!("buffer map channel closed"))?;
        map_result?;

        let mapped = slice.get_mapped_range();
        let width_bytes = (self.width as usize) * 4;
        let mut rgba = vec![0u8; (self.width as usize) * (self.height as usize) * 4];

        for row in 0..(self.height as usize) {
            let src_start = row * (self.bytes_per_row as usize);
            let src_end = src_start + width_bytes;
            let dst_start = row * width_bytes;
            let dst_end = dst_start + width_bytes;
            rgba[dst_start..dst_end].copy_from_slice(&mapped[src_start..src_end]);
        }

        drop(mapped);
        self.staging_buffer.unmap();
        Ok(rgba)
    }
}

fn focal_to_fov(focal: f32, pixels: f32) -> Rad<f32> {
    Rad(2.0 * (pixels / (2.0 * focal)).atan())
}

#[pollster::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    let opt = Opt::parse();

    let mut server = StreamRenderer::new(&opt).await?;

    // Emit resolution once so the Python side can validate stream configuration.
    {
        let stdout = std::io::stdout();
        let mut writer = stdout.lock();
        writer
            .write_all(format!("READY {} {}\n", opt.width, opt.height).as_bytes())
            .context("failed to write ready line")?;
        writer.flush().context("failed to flush ready line")?;
    }

    let stdin = std::io::stdin();
    let mut lines = BufReader::new(stdin.lock()).lines();

    while let Some(line) = lines.next() {
        let line = line.context("failed reading stdin line")?;
        let trimmed = line.trim();

        if trimmed.is_empty() {
            continue;
        }
        if trimmed.eq_ignore_ascii_case("quit") {
            break;
        }

        let stdout = std::io::stdout();
        let mut writer = stdout.lock();

        let frame_result = match parse_camera(trimmed, server.width, server.height, server.fx, server.fy) {
            Ok(camera) => server.render_camera(camera).await,
            Err(err) => {
                log::error!("invalid pose '{}': {}", trimmed, err);
                Err(err)
            }
        };

        match frame_result {
            Ok(frame) => {
                writer
                    .write_all(&(frame.len() as u32).to_le_bytes())
                    .context("failed to write frame byte-size header")?;
                writer
                    .write_all(&frame)
                    .context("failed to write frame payload")?;
            }
            Err(err) => {
                log::error!("render failed for '{}': {}", trimmed, err);
                writer
                    .write_all(&0u32.to_le_bytes())
                    .context("failed to write error frame header")?;
            }
        }
        writer.flush().context("failed to flush frame")?;
    }

    Ok(())
}