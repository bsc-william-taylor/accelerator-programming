
const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

kernel void unsharp_mask_pass_one(
    read_only image2d_t input, 
    write_only image2d_t output,
    constant float* hori,
    const int radius
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
    constant float* vert,
    const int radius
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