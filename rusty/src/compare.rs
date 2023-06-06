use std::error::Error;
use std::ffi::CString;

use rustacuda::prelude::*;
use rustacuda::memory::DeviceBox;
use rustacuda::launch::function;
use rustacuda::function::{BlockSize, GridSize};
use rustacuda::memory::DeviceBuffer;

pub fn cosine_similarity_cpu(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (magnitude_a * magnitude_b)
}


pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, Box<dyn Error>>{
    rustacuda::init(CudaFlags::empty())?;

    let a_device = DeviceBuffer::from_slice(&a)?;
    let b_device = DeviceBuffer::from_slice(&b)?;

    //intermediate results storage
    let dot_product_device = DeviceBox::new(&[0.0])?;
    let magnitude_a_device = DeviceBox::new(&[0.0])?;
    let magnitude_b_device = DeviceBox::new(&[0.0])?;

    //Create a cuda module
    let module_data = CString::new(include_str!("cosine_similarity.ptx"))?;
    let module = Module::load_from_string(&module_data)?;

    let function_name = CString::new("cosine_similarity")?;
    let function = module.get_function(&function_name)?;

    let block_size = BlockSize {
        x: 256,
        y: 1,
        z: 1
    };

    let grid_size = GridSize {
        x: (a.len() as f32 / block_size.x as f32).ceil() as u32,
        y: 1,
        z: 1
    };

    //launch kernel
    unsafe {
        function.launch(
            &[
                &a_device.as_device_ptr(),
                &b_device.as_device_ptr(),
                &dot_product_device.as_device_ptr(),
                &magnitude_a_device.as_device_ptr(),
                &magnitude_b_device.as_device_ptr(),
                &(a.len() as i32)
            ],
            &block_size,
            &grid_size,
            None
        )?;
    }


    // Copy the results back to the host
    let mut dot_product_host = [0.0];
    let mut magnitude_a_host = [0.0];
    let mut magnitude_b_host = [0.0];
    dot_product_device.copy_to(&mut dot_product_host)?;
    magnitude_a_device.copy_to(&mut magnitude_a_host)?;
    magnitude_b_device.copy_to(&mut magnitude_b_host)?;
    
    // Perform the final computation on the host
    
    let cosine_similarity = dot_product_host[0] / (magnitude_a_host[0] * magnitude_b_host[0]);
    
    Ok(cosine_similarity)
}


