
kernel void copy(write_only image2d_t image)
{
    sampler_t sampler =  CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    const int x = get_global_id(0);
    const int y = get_global_id(1);

    //write_imagef(image, (int2)(x, y), (float4)(1.0, 0.0, 0.0, 1.0));//read_imagef(input, sampler, (int2)(x, y)));
}

