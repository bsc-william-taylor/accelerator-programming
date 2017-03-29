
/*
    When not passed as a command line 
    option defaults are used
*/
#ifndef alpha
    #define alpha 1.5
    #define beta -0.5
    #define gamma 0.0
    #define radius 5
    #define masksize 0
#endif

/*
    Blur Mask as constant memory if it will fit in constant
    memory usally 64kb, else global.
*/
#if masksize <= 64000 // 64k
    #define mask_storage constant
#else 
    #define mask_storage global
#endif

const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

kernel void unsharp_mask_pass_one(
    read_only image2d_t input, 
    write_only image2d_t output,
    mask_storage float* hori
)
{   
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    float4 blurred = (float4)0.0f;

    for(int i = -radius, index = 0; i <= radius; ++i)
    {
        blurred += read_imagef(input, sampler, (int2)(x + i, y)) * (float4)hori[index++];
    }

    const float4 colour = radius != 0 ? blurred : read_imagef(input, sampler, (int2)(x, y));

    write_imagef(output, (int2)(x, y), colour);
}

kernel void unsharp_mask_pass_two(
    read_only image2d_t image,
    read_only image2d_t input, 
    write_only image2d_t output,
    mask_storage float* vert
)
{ 
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const float4 colour = read_imagef(image, sampler, (int2)(x, y));   

    float4 blurred = (float4)0.0f;

    for(int i = -radius, index = 0; i <= radius; ++i)
    {
        blurred += read_imagef(input, sampler, (int2)(x, y + i)) * (float4)vert[index++];
    }

    const float4 sharp = (float4)(colour * (float4)alpha + blurred * (float4)beta + (float4)gamma);

    write_imagef(output, (int2)(x, y), radius == 0 ? colour : sharp);
}

kernel void unsharp_mask(
    read_only image2d_t in, 
    write_only image2d_t out, 
    mask_storage float* mask, 
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
    const float4 sharp = (float4)(colour * (float4)alpha + blurred * (float4)beta + (float4)gamma);

    write_imagef(out, px, radius == 0 ? colour : sharp);
}

kernel void unsharp_mask_sections(
    read_only image2d_t input, 
    write_only image2d_t output,
    mask_storage float* mask,
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
    mask_storage float* mask
)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    unsharp_mask(input, output, mask, (int2)(x, y));
}