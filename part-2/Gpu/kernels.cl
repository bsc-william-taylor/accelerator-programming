
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
    const float PI = 3.14159265359f;
    float4 blurred = (float4)0.0f;
    int rs = ceil(radius * 2.57);
    float sum = 0.0;

    for(int iy = pixel.y-rs; iy <= pixel.y+rs; ++iy) { 
        for(int ix = pixel.x-rs; ix <= pixel.x+rs; ++ix) {
            float distance = (ix - pixel.x) * (ix - pixel.x) + (iy - pixel.y) * (iy - pixel.y);
            float weight = exp(-distance / (2 * radius * radius)) / (PI * 2 * radius * radius);
		    blurred += read_imagef(input, sampler, (int2)(ix, iy)) * weight;
            sum += weight;
        }
     }

    blurred /= sum;

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
    const int x = get_global_id(0) + offsetX;
    const int y = get_global_id(1) + offsetY;

    unsharp_mask(input, output, mask, radius, (int2)(x, y));
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