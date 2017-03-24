
kernel void copy(read_only image2d_t input, write_only image2d_t ouput, const int radius)
{
    sampler_t sampler =  CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    const int x = get_global_id(0);
    const int y = get_global_id(1);

    float4 colour = read_imagef(input, sampler, (int2)(x, y));

    write_imagef(ouput, (int2)(x, y), radius == 0 ? colour : (float4)(1.0, 0.0, 0.0, 1.0));
}

