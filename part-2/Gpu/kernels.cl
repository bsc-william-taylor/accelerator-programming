
/*
    When not passed as a command line 
    option defaults are used
*/
#ifndef alpha
    #define alpha 1.5
    #define beta -0.5
    #define gamma 0.0
#endif

const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

kernel void unsharp_mask(
    read_only image2d_t input, 
    write_only image2d_t output, 
    constant float* mask, 
    const int radius,
    const int2 pixel
)
{
    const float4 colour = read_imagef(input, sampler, pixel);

    float4 blurred = (float4)0.0f;
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

    const float4 sharpColour = (float4)(colour * alpha + blurred * beta + gamma);

    write_imagef(output, pixel, radius == 0 ? colour : sharpColour);
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