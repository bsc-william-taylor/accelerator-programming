
/*
    When not passed as a command line 
    option defaults are used
*/
#ifndef alpha
    #define alpha 1.5
    #define beta -0.5
    #define gamma 0.0
    #define radius 5
#endif

#if (((radius*2+1) * (radius*2+1)) * 4) < 64000
    #define MASK_MEMORY global
#else 
    #define MASK_MEMORY global
#endif

const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

kernel void unsharp_mask(
    read_only image2d_t in, 
    write_only image2d_t out, 
    MASK_MEMORY float* mask, 
    const int2 px
)
{
    float4 blurred = (float4)0.0f;
    int index = 0;

    for(int y = -radius; y <= radius; ++y) 
    { 
        for(int x = -radius; x <= radius; ++x)
        {
		    blurred += read_imagef(in, sampler, (int2)(px.x + x, px.y + y)) * mask[index++];
        }
     }

    const float4 colour = read_imagef(in, sampler, px);   
    const float4 sharpColour = (float4)(colour * alpha + blurred * beta + gamma);

    write_imagef(out, px, radius == 0 ? colour : sharpColour);
}

kernel void unsharp_mask_sections(
    read_only image2d_t input, 
    write_only image2d_t output, 
    MASK_MEMORY float* mask, 
    const int offsetX,
    const int offsetY
)
{
    const int x = get_global_id(0) + offsetX;
    const int y = get_global_id(1) + offsetY;

    unsharp_mask(input, output, mask, (int2)(x, y));
}

kernel void unsharp_mask_full(
    read_only image2d_t input, 
    write_only image2d_t output, 
    MASK_MEMORY float* mask
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    unsharp_mask(input, output, mask, (int2)(x, y));
}