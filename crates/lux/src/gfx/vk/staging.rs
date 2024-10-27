use crate::prelude::*;

use super::include::*;

pub struct StagedWrite {
    buffer: Buffer,
    from: DeviceSize,
    to: DeviceSize,
    size: DeviceSize,
}

#[derive(Resource)]
pub struct StagingBuffer {
    buffer: Buffer,
    memory: Memory,
    writes: Vec<StagedWrite>,
    waiting: Vec<u8>,
    size: u64,
    cursor: u64,
    last_write: u64,
}

impl StagingBuffer {
    pub(crate) fn create(
        physical_device: ResMut<PhysicalDevice>,
        device: ResMut<Device>,
    ) -> Insert<Self> {
        let size = 1_000_000;
        let buffer = Buffer::create(*device, size, BufferUsageFlagBits::TransferSrc as u32);

        let req = buffer.requirements(*device);

        let memory = Memory::host(*physical_device, *device, req).bind_buffer(*device, buffer);

        (Self {
            buffer,
            memory,
            size,
            cursor: 0,
            last_write: 0,
            writes: vec![],
            waiting: vec![],
        })
        .into()
    }
    pub(crate) fn write<T: Copy>(&mut self, buffer: Buffer, offset: u64, data: T) {
        let data = unsafe {
            ptr::slice_from_raw_parts(&data as *const _ as *const u8, mem::size_of::<T>())
                .as_ref()
                .unwrap()
        };
        self.writes.push(StagedWrite {
            to: offset,
            from: self.cursor,
            size: data.len() as u64,
            buffer,
        });
        self.waiting.extend(data);
        if self.cursor >= self.size {
            self.cursor = 0;
        }
        self.cursor += data.len() as u64;
    }
    pub(crate) fn host_transfer(&mut self, device: Device) {
        let data = self
            .memory
            .map(device, self.last_write, self.waiting.len() as u64) as *mut _;
        unsafe { ptr::copy(self.waiting.as_ptr(), data, self.waiting.len()) };
        self.memory.unmap(device);
        self.waiting.clear();
    }
    pub(crate) fn device_copy(&mut self, cmd: CommandBuffer) {
        cmd.pipeline_barrier_buffer(
            PipelineStageFlagBits::TopOfPipe as u32,
            PipelineStageFlagBits::Transfer as u32,
            &[BufferBarrier {
                src_access: 0,
                dst_access: AccessFlagBits::TransferRead as u32,
                buffer: self.buffer,
                offset: self.last_write,
                size: self.writes.iter().map(|x| x.size).sum(),
            }],
        );
        for write in self.writes.drain(..) {
            cmd.copy_buffer(self.buffer, write.buffer, write.from, write.to, write.size);
        }
        if self.cursor >= self.size {
            self.cursor = 0;
        }
        self.last_write = self.cursor;
    }
}
