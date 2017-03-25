
const sampler_t sampler =  CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

kernel void unsharp_mask(
    read_only image2d_t input, 
    write_only image2d_t output, 
    constant float* mask, 
    const int radius,
    const int2 pixel
)
{
    const float alpha = 1.5, beta = -0.5, gamma = 0.0;
    const float4 colour = read_imagef(input, sampler, pixel);

    float4 blurred = (float4)0.0;
    int maskIndex = 0;

    for (int i = -radius; i <= radius; ++i)
    {
	    for (int j = -radius; j <= radius; ++j)
	    {				
            const int2 location = (int2)(pixel.x + j, pixel.y + i);
		    blurred += read_imagef(input, sampler, location) * mask[maskIndex];
            maskIndex++;
	    }
    }

    write_imagef(output, pixel, radius == 0 ? colour : (float4)(colour * alpha + blurred * beta + gamma));
}

kernel void unsharp_mask_sections(
    read_only image2d_t input, 
    write_only image2d_t output, 
    constant float* mask, 
    const int radius,
    const int offsetX,
    const int offsetY
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    unsharp_mask(input, output, mask, radius, (int2)(x + offsetX, y + offsetY));
}

kernel void unsharp_mask_full(
    read_only image2d_t input, 
    write_only image2d_t output, 
    constant float* mask, 
    const int radius)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    unsharp_mask(input, output, mask, radius, (int2)(x, y));
}