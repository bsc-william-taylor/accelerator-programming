
kernel void copy(read_only image2d_t input, write_only image2d_t output)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    float4 colour = read_imagef(input, sampler, (int2)(x, y));
    write_imagef(output, (int2)(x, y), colour);
}

kernel void add_weighted(read_only image2d_t input, read_only image2d_t blur, write_only image2d_t output)
{ 
    sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    int2 px = (int2)(get_global_id(0), get_global_id(1));

    float alpha = 1.5, beta = -0.5, gamma = 0.0;
    float4 colour = read_imagef(input, sampler, px);
    float4 blurred = read_imagef(blur, sampler, px);

    float r = colour.x * alpha + blurred.x * beta + gamma;
    float g = colour.y * alpha + blurred.y * beta + gamma;
    float b = colour.z * alpha + blurred.z * beta + gamma;

    write_imagef(output, px, (float4)(r, g, b, colour.w));
}

kernel void gaussian_blur(read_only image2d_t input, write_only image2d_t output, int radius)
{ 
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    // TODO: Implement full gaussian blur
}

