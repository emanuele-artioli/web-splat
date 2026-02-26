/*
    This file implements a gpu version of radix sort. A good introduction to general purpose radix sort can
    be found here: http://www.codercorner.com/RadixSortRevisited.htm

    The gpu radix sort implemented here is a reimplementation of the vulkan radix sort found in the fuchsia repos: https://fuchsia.googlesource.com/fuchsia/+/refs/heads/main/src/graphics/lib/compute/radix_sort/
    Currently only the sorting for floating point key-value pairs is implemented, as only this is needed for this project

    All shaders can be found in shaders/radix_sort.wgsl
*/

use wgpu::{ComputePassDescriptor, util::DeviceExt};

const HISTOGRAM_WG_SIZE: usize = 256;
const PREFIX_WG_SIZE: usize = 1 << 7;
const SCATTER_WG_SIZE: usize = 256;
const PASSES:usize = 4; // only 4 passes are needed for 32 bit keys with a radix of 8, as 4*8 = 32

const RS_RADIX_LOG2: usize = 8;
const RS_RADIX_SIZE: usize = 1 << RS_RADIX_LOG2;
const RS_KEYVAL_SIZE: usize = 32 / RS_RADIX_LOG2;
const RS_HISTOGRAM_BLOCK_ROWS: usize = 15;
const RS_SCATTER_BLOCK_ROWS: usize = RS_HISTOGRAM_BLOCK_ROWS;

pub struct GPURSSorter {
    bind_group_layout: wgpu::BindGroupLayout,
    render_bind_group_layout: wgpu::BindGroupLayout,
    preprocess_bind_group_layout: wgpu::BindGroupLayout,
    zero_p: wgpu::ComputePipeline,
    histogram_p: wgpu::ComputePipeline,
    prefix_p: wgpu::ComputePipeline,
    scatter_even_p: wgpu::ComputePipeline,
    scatter_odd_p: wgpu::ComputePipeline,
}

pub struct PointCloudSortStuff {
    pub num_points: usize,
    pub(crate) sorter_uni: wgpu::Buffer, // uniform buffer information
    pub(crate) sorter_dis: wgpu::Buffer, // dispatch buffer
    pub(crate) sorter_bg: wgpu::BindGroup, // sorter bind group
    pub(crate) sorter_render_bg: wgpu::BindGroup, // bind group only with the sorted indices for rendering
    pub(crate) sorter_bg_pre: wgpu::BindGroup, // bind group for the preprocess (is the sorter_dis and sorter_bg merged as we only have a limited amount of bgs for the preprocessing)
}

#[allow(dead_code)]
pub struct IndirectDispatch {
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
}

pub struct GeneralInfo {
    pub keys_size: u32,
    pub padded_size: u32,
    pub even_pass: u32,
    pub odd_pass: u32,
    _pad:u32,
}

unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::core::slice::from_raw_parts((p as *const T) as *const u8, ::core::mem::size_of::<T>())
}

impl GPURSSorter {
    pub fn create_sort_stuff(
        &self,
        device: &wgpu::Device,
        num_points: usize,
    ) -> PointCloudSortStuff {
        let (sorter_b_a, sorter_b_b, sorter_p_a, sorter_p_b) =
            GPURSSorter::create_keyval_buffers(device, num_points, 4);
        let sorter_int = self.create_internal_mem_buffer(device, num_points);
        let (sorter_uni, sorter_dis, sorter_bg) = self.create_bind_group(
            device,
            num_points,
            &sorter_int,
            &sorter_b_a,
            &sorter_b_b,
            &sorter_p_a,
            &sorter_p_b,
        );
        let sorter_render_bg = self.create_bind_group_render(device, &sorter_uni, &sorter_p_a);
        let sorter_bg_pre = self.create_bind_group_preprocess(
            device,
            &sorter_uni,
            &sorter_dis,
            &sorter_b_a,
            &sorter_p_a,
        );

        PointCloudSortStuff {
            num_points,
            sorter_uni,
            sorter_dis,
            sorter_bg,
            sorter_render_bg,
            sorter_bg_pre,
        }
    }

    pub fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = Self::bind_group_layouts(device);
        let render_bind_group_layout = Self::bind_group_layout_rendering(device);
        let preprocess_bind_group_layout = Self::bind_group_layout_preprocess(device);

        let pipeline_layout: wgpu::PipelineLayout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("radix sort pipeline layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Radix sort shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/radix_sort.wgsl").into()),
        });

        let override_constants: Vec<(String, f64)> = vec![
            ("histogram_wg_size".to_string(), HISTOGRAM_WG_SIZE as f64),
            ("prefix_wg_size".to_string(), PREFIX_WG_SIZE as f64),
            ("scatter_wg_size".to_string(), SCATTER_WG_SIZE as f64),
            ("rs_keyval_size".to_string(), RS_KEYVAL_SIZE as f64),
        ];
        let override_refs: Vec<(&str, f64)> = override_constants
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        let compilation_options = wgpu::PipelineCompilationOptions {
            constants: &override_refs,
            ..Default::default()
        };
        let zero_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Zero the histograms"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("zero_histograms"),
            compilation_options: compilation_options.clone(),
            cache: None,
        });
        let histogram_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("calculate_histogram"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("calculate_histogram"),
            compilation_options: compilation_options.clone(),
            cache: None,
        });
        let prefix_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("prefix_histogram"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("prefix_histogram"),
            compilation_options: compilation_options.clone(),
            cache: None,
        });
        let scatter_even_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scatter_even"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scatter_even"),
            compilation_options: compilation_options.clone(),
            cache: None,
        });
        let scatter_odd_p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scatter_odd"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scatter_odd"),
            compilation_options,
            cache: None,
        });

        return Self {
            bind_group_layout,
            render_bind_group_layout,
            preprocess_bind_group_layout,
            zero_p,
            histogram_p,
            prefix_p,
            scatter_even_p,
            scatter_odd_p,
        };
    }


    // layouts used by the sorting pipeline, as the dispatch buffer has to be in separate bind group
    pub fn bind_group_layouts(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Radix bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    }
    // is used by the preprocess pipeline as the limitation of bind groups forces us to only use 1 bind group for the sort infos
    pub fn bind_group_layout_preprocess(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Radix bind group layout for preprocess pipeline"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    }
    // used by the renderer, as read_only : false is not allowed without an extension
    pub fn bind_group_layout_rendering(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Radix bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
    }

    fn get_scatter_histogram_sizes(keysize: usize) -> (usize, usize, usize, usize, usize, usize) {
        // as a general rule of thumb, scater_blocks_ru is equal to histo_blocks_ru, except the amount of elements in these two stages is different

        let scatter_block_kvs = HISTOGRAM_WG_SIZE * RS_SCATTER_BLOCK_ROWS;
        let scatter_blocks_ru = (keysize + scatter_block_kvs - 1) / scatter_block_kvs;
        let count_ru_scatter = scatter_blocks_ru * scatter_block_kvs;

        let histo_block_kvs = HISTOGRAM_WG_SIZE * RS_HISTOGRAM_BLOCK_ROWS;
        let histo_blocks_ru = (count_ru_scatter + histo_block_kvs - 1) / histo_block_kvs;
        let count_ru_histo = histo_blocks_ru * histo_block_kvs;

        return (
            scatter_block_kvs,
            scatter_blocks_ru,
            count_ru_scatter,
            histo_block_kvs,
            histo_blocks_ru,
            count_ru_histo,
        );
    }

    pub fn create_keyval_buffers(
        device: &wgpu::Device,
        keysize: usize,
        bytes_per_payload_elem: usize,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        //let (_, _, _, _, _, count_ru_histo) = Self::get_scatter_histogram_sizes(keysize);
        let keys_per_workgroup = HISTOGRAM_WG_SIZE * RS_HISTOGRAM_BLOCK_ROWS;
        let count_ru_histo =
            ((keysize + keys_per_workgroup) / keys_per_workgroup + 1) * keys_per_workgroup;

        // creating the two needed buffers for sorting
        let buffer_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix data buffer a"),
            size: (count_ru_histo * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buffer_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix data buffer a"),
            size: (count_ru_histo * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        assert!(bytes_per_payload_elem == 4); // currently only 4 byte values are allowed
        let payload_size = (keysize * bytes_per_payload_elem).max(1); // make sure that we have at least 1 byte of data;
        let payload_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix data buffer a"),
            size: payload_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let payload_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Radix data buffer a"),
            size: payload_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        return (buffer_a, buffer_b, payload_a, payload_b);
    }

    // caclulates and allocates a buffer that is sufficient for holding all needed information for
    // sorting. This includes the histograms and the temporary scatter buffer
    // @return: tuple containing [internal memory buffer (should be bound at shader binding 1, count_ru_histo (padded size needed for the keyval buffer)]
    pub fn create_internal_mem_buffer(
        &self,
        device: &wgpu::Device,
        keysize: usize,
    ) -> wgpu::Buffer {
        // The "internal" memory map looks like this:
        //   +---------------------------------+ <-- 0
        //   | histograms[keyval_size]         |
        //   +---------------------------------+ <-- keyval_size                           * histo_size
        //   | partitions[scatter_blocks_ru-1] |
        //   +---------------------------------+ <-- (keyval_size + scatter_blocks_ru - 1) * histo_size
        //   | workgroup_ids[keyval_size]      |
        //   +---------------------------------+ <-- (keyval_size + scatter_blocks_ru - 1) * histo_size + workgroup_ids_size

        let (_, scatter_blocks_ru, _, _, _, _) = Self::get_scatter_histogram_sizes(keysize);

        let histo_size = RS_RADIX_SIZE * std::mem::size_of::<u32>();

        let internal_size = (RS_KEYVAL_SIZE + scatter_blocks_ru - 1 + 1) * histo_size; // +1 safety

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Internal radix sort buffer"),
            size: internal_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        return buffer;
    }

    pub fn create_bind_group(
        &self,
        device: &wgpu::Device,
        keysize: usize,
        internal_mem_buffer: &wgpu::Buffer,
        keyval_a: &wgpu::Buffer,
        keyval_b: &wgpu::Buffer,
        payload_a: &wgpu::Buffer,
        payload_b: &wgpu::Buffer,
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::BindGroup) {
        let (_, scatter_blocks_ru, _, _, _, count_ru_histo) =
            Self::get_scatter_histogram_sizes(keysize);

        let dispatch_infos = IndirectDispatch {
            dispatch_x: scatter_blocks_ru as u32,
            dispatch_y: 1,
            dispatch_z: 1,
        };
        let uniform_infos = GeneralInfo {
            keys_size: keysize as u32,
            padded_size: count_ru_histo as u32,
            even_pass: 0,
            odd_pass: 0,
            _pad: 0,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Radix uniform buffer"),
            contents: unsafe { any_as_u8_slice(&uniform_infos) },
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let dispatch_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dispatch indirect buffer"),
            contents: unsafe { any_as_u8_slice(&dispatch_infos) },
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::INDIRECT,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Radix bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: internal_mem_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: keyval_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: keyval_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: payload_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: payload_b.as_entire_binding(),
                },
            ],
        });
        return (uniform_buffer, dispatch_buffer, bind_group);
    }
    pub fn create_bind_group_render(
        &self,
        device: &wgpu::Device,
        general_infos: &wgpu::Buffer,
        payload_a: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let rendering_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Render bind group"),
            layout: &self.render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: general_infos.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: payload_a.as_entire_binding(),
                },
            ],
        });
        return rendering_bind_group;
    }
    pub fn create_bind_group_preprocess(
        &self,
        device: &wgpu::Device,
        uniform_buffer: &wgpu::Buffer,
        dispatch_buffer: &wgpu::Buffer,
        keyval_a: &wgpu::Buffer,
        payload_a: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Preprocess bind group"),
            layout: &self.preprocess_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: keyval_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: payload_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dispatch_buffer.as_entire_binding(),
                },
            ],
        });
        return bind_group;
    }

    pub fn record_reset_indirect_buffer(
        indirect_buffer: &wgpu::Buffer,
        uniform_buffer: &wgpu::Buffer,
        queue: &wgpu::Queue,
    ) {
        queue.write_buffer(indirect_buffer, 0, &[0u8, 0u8, 0u8, 0u8]); // nulling dispatch x
        queue.write_buffer(uniform_buffer, 0, &[0u8, 0u8, 0u8, 0u8]); // nulling keysize
    }

    pub fn record_calculate_histogram(
        &self,
        bind_group: &wgpu::BindGroup,
        keysize: usize,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // histogram has to be zeroed out such that counts that might have been done in the past are erased and do not interfere with the new count
        // encoder.clear_buffer(histogram_buffer, 0, None);

        // as we only deal with 32 bit float values always 4 passes are conducted
        let (_, _, _, _, hist_blocks_ru, _) = Self::get_scatter_histogram_sizes(keysize);
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("zeroing the histogram"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.zero_p);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(hist_blocks_ru as u32, 1, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("calculate histogram"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.histogram_p);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(hist_blocks_ru as u32, 1, 1);
        }
    }
    pub fn record_calculate_histogram_indirect(
        &self,
        bind_group: &wgpu::BindGroup,
        dispatch_buffer: &wgpu::Buffer,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("zeroing the histogram"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.zero_p);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups_indirect(dispatch_buffer, 0);
        }

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("calculate histogram"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.histogram_p);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups_indirect(dispatch_buffer, 0);
        }
    }

    // There does not exist an indirect histogram dispatch as the number of prefixes is determined by the amount of passes
    pub fn record_prefix_histogram(
        &self,
        bind_group: &wgpu::BindGroup,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("prefix histogram"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.prefix_p);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(PASSES as u32, 1, 1);
    }

    pub fn record_scatter_keys(
        &self,
        bind_group: &wgpu::BindGroup,
        keysize: usize,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let (_, scatter_blocks_ru, _, _, _, _) = Self::get_scatter_histogram_sizes(keysize);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Scatter keyvals"),
            timestamp_writes: None,
        });

        pass.set_bind_group(0, bind_group, &[]);
        pass.set_pipeline(&self.scatter_even_p);
        pass.dispatch_workgroups(scatter_blocks_ru as u32, 1, 1);

        pass.set_pipeline(&self.scatter_odd_p);
        pass.dispatch_workgroups(scatter_blocks_ru as u32, 1, 1);

        pass.set_pipeline(&self.scatter_even_p);
        pass.dispatch_workgroups(scatter_blocks_ru as u32, 1, 1);

        pass.set_pipeline(&self.scatter_odd_p);
        pass.dispatch_workgroups(scatter_blocks_ru as u32, 1, 1);
    }
    pub fn record_scatter_keys_indirect(
        &self,
        bind_group: &wgpu::BindGroup,
        dispatch_buffer: &wgpu::Buffer,
        encoder: &mut wgpu::CommandEncoder,
    ) {

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Scatter keyvals"),
            timestamp_writes: None,
        });

        pass.set_bind_group(0, bind_group, &[]);
        pass.set_pipeline(&self.scatter_even_p);
        pass.dispatch_workgroups_indirect(dispatch_buffer, 0);

        pass.set_pipeline(&self.scatter_odd_p);
        pass.dispatch_workgroups_indirect(dispatch_buffer, 0);

        pass.set_pipeline(&self.scatter_even_p);
        pass.dispatch_workgroups_indirect(dispatch_buffer, 0);

        pass.set_pipeline(&self.scatter_odd_p);
        pass.dispatch_workgroups_indirect(dispatch_buffer, 0);
    }

    pub fn record_sort(
        &self,
        bind_group: &wgpu::BindGroup,
        keysize: usize,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.record_calculate_histogram(&bind_group, keysize, encoder);
        self.record_prefix_histogram(&bind_group, encoder);
        self.record_scatter_keys(&bind_group, keysize, encoder);
    }
    pub fn record_sort_indirect(
        &self,
        bind_group: &wgpu::BindGroup,
        dispatch_buffer: &wgpu::Buffer,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.record_calculate_histogram_indirect(bind_group, dispatch_buffer, encoder);
        self.record_prefix_histogram(bind_group, encoder);
        self.record_scatter_keys_indirect(bind_group,  dispatch_buffer, encoder);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a wgpu device and queue for testing.
    async fn create_test_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No GPU adapter found");

        let adapter_limits = adapter.limits();
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::SUBGROUP,
                required_limits: wgpu::Limits {
                    max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
                    max_storage_buffers_per_shader_stage: 12,
                    max_compute_workgroup_storage_size: 1 << 15,
                    ..adapter_limits
                },
                label: Some("test device"),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .expect("Failed to create device")
    }

    /// Runs a GPU radix sort on the given input keys and returns the sorted keys and payloads.
    async fn run_sort(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sorter: &GPURSSorter,
        input_keys: &[f32],
        input_payloads: &[u32],
    ) -> (Vec<f32>, Vec<u32>) {
        let n = input_keys.len();
        assert_eq!(n, input_payloads.len());

        let internal_mem_buffer = sorter.create_internal_mem_buffer(device, n);
        let (keyval_a, keyval_b, payload_a, payload_b) =
            GPURSSorter::create_keyval_buffers(device, n, 4);
        let (_uniform_buffer, _dispatch_buffer, bind_group) = sorter.create_bind_group(
            device,
            n,
            &internal_mem_buffer,
            &keyval_a,
            &keyval_b,
            &payload_a,
            &payload_b,
        );

        upload_to_buffer(&keyval_a, device, queue, input_keys);
        upload_to_buffer(&payload_a, device, queue, input_payloads);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("test sort encoder"),
        });
        sorter.record_sort(&bind_group, n, &mut encoder);
        let idx = queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: None,
            })
            .unwrap();

        let sorted_keys = download_buffer::<f32>(&keyval_a, device, queue).await;
        let sorted_payloads = download_buffer::<u32>(&payload_a, device, queue).await;
        (sorted_keys[..n].to_vec(), sorted_payloads[..n].to_vec())
    }

    #[test]
    fn test_radix_sort_various_sizes() {
        // keys_per_workgroup = HISTOGRAM_WG_SIZE * RS_HISTOGRAM_BLOCK_ROWS = 256 * 15 = 3840
        // Test sizes that exercise different code paths:
        // - small sizes (< 1 workgroup)
        // - exactly 1 workgroup
        // - sizes crossing workgroup boundaries
        // - multiple workgroups
        // - large sizes
        let test_sizes: Vec<usize> = vec![
            1,     // single element
            2,     // minimal
            7,     // small odd
            16,    // small power of 2
            255,   // just under wg size
            256,   // exactly wg size (HISTOGRAM_WG_SIZE)
            257,   // just over wg size
            1000,  // medium, not aligned
            3839,  // one less than keys_per_workgroup
            3840,  // exactly keys_per_workgroup
            3841,  // one more than keys_per_workgroup
            4096,  // power of 2
            7680,  // exactly 2 workgroups
            8192,  // same as existing test_sort
            10000, // large non-aligned
            15360, // 4 workgroups
            16384, // large power of 2
            50000, // stress test
        ];

        pollster::block_on(async {
            let (device, queue) = create_test_device().await;
            let sorter = GPURSSorter::new(&device);

            for &n in &test_sizes {
                // Create keys in reverse order
                let input_keys: Vec<f32> = (0..n).rev().map(|x| x as f32).collect();
                // Payload tracks original index
                let input_payloads: Vec<u32> = (0..n).rev().map(|x| x as u32).collect();
                let expected_keys: Vec<f32> = (0..n).map(|x| x as f32).collect();

                let (sorted_keys, sorted_payloads) =
                    run_sort(&device, &queue, &sorter, &input_keys, &input_payloads).await;

                for i in 0..n {
                    assert_eq!(
                        sorted_keys[i], expected_keys[i],
                        "Key mismatch at index {i} for size {n}: got {} expected {}",
                        sorted_keys[i], expected_keys[i]
                    );
                }

                // Verify payloads were permuted consistently with keys
                for i in 0..n {
                    let key = sorted_keys[i] as u32;
                    assert_eq!(
                        sorted_payloads[i], key,
                        "Payload mismatch at index {i} for size {n}: payload {} but key {}",
                        sorted_payloads[i], key
                    );
                }

                eprintln!("  sort size {n}: OK");
            }
        });
    }

    #[test]
    fn test_radix_sort_already_sorted() {
        pollster::block_on(async {
            let (device, queue) = create_test_device().await;
            let sorter = GPURSSorter::new(&device);

            let n = 8192;
            let input_keys: Vec<f32> = (0..n).map(|x| x as f32).collect();
            let input_payloads: Vec<u32> = (0..n).map(|x| x as u32).collect();

            let (sorted_keys, sorted_payloads) =
                run_sort(&device, &queue, &sorter, &input_keys, &input_payloads).await;

            for i in 0..n {
                assert_eq!(
                    sorted_keys[i], i as f32,
                    "Already-sorted key mismatch at index {i}"
                );
                assert_eq!(
                    sorted_payloads[i], i as u32,
                    "Already-sorted payload mismatch at index {i}"
                );
            }
        });
    }

    #[test]
    fn test_radix_sort_all_same_keys() {
        pollster::block_on(async {
            let (device, queue) = create_test_device().await;
            let sorter = GPURSSorter::new(&device);

            let n = 4096;
            let input_keys: Vec<f32> = vec![42.0; n];
            let input_payloads: Vec<u32> = (0..n as u32).collect();

            let (sorted_keys, _sorted_payloads) =
                run_sort(&device, &queue, &sorter, &input_keys, &input_payloads).await;

            for i in 0..n {
                assert_eq!(sorted_keys[i], 42.0, "Same-key value changed at index {i}");
            }
        });
    }

    #[test]
    fn test_radix_sort_random_data() {
        use rand::RngExt;
        pollster::block_on(async {
            let (device, queue) = create_test_device().await;
            let sorter = GPURSSorter::new(&device);

            let mut rng = rand::rng();
            let sizes = [100, 1024, 5000, 8192, 12345];

            for &n in &sizes {
                let input_keys: Vec<f32> = (0..n).map(|_| rng.random_range(0.0f32..1e6)).collect();
                let input_payloads: Vec<u32> = (0..n as u32).collect();

                let (sorted_keys, sorted_payloads) =
                    run_sort(&device, &queue, &sorter, &input_keys, &input_payloads).await;

                // Verify sorted order
                for i in 1..n {
                    assert!(
                        sorted_keys[i] >= sorted_keys[i - 1],
                        "Random sort not monotonic at index {i} for size {n}: {} > {}",
                        sorted_keys[i - 1],
                        sorted_keys[i]
                    );
                }

                // Verify payload consistency: the payload should reference back
                // to the original key value
                for i in 0..n {
                    let orig_idx = sorted_payloads[i] as usize;
                    assert_eq!(
                        sorted_keys[i], input_keys[orig_idx],
                        "Payload-key mismatch at index {i} for size {n}"
                    );
                }

                eprintln!("  random sort size {n}: OK");
            }
        });
    }

    fn upload_to_buffer<T: bytemuck::Pod>(
        buffer: &wgpu::Buffer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        values: &[T],
    ) {
        let staging_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Staging buffer"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copye endoder"),
        });
        encoder.copy_buffer_to_buffer(&staging_buffer, 0, buffer, 0, staging_buffer.size());
        let idx = queue.submit([encoder.finish()]);

        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: None,
            })
            .unwrap();
        staging_buffer.destroy();
    }

    async fn download_buffer<T: Clone>(
        buffer: &wgpu::Buffer,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Vec<T> {
        // copy buffer data
        let download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Download buffer"),
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &download_buffer, 0, buffer.size());
        let idx = queue.submit([encoder.finish()]);

        // download buffer
        let buffer_slice = download_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: None,
            })
            .unwrap();
        rx.receive().await.unwrap().unwrap();
        let data = buffer_slice.get_mapped_range();
        let r;

        unsafe {
            let (_, d, _) = data.align_to::<T>();
            r = d.to_vec();
        }

        return r;
    }
}
