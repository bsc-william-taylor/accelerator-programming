
kernel void unsharp_mask(read_only image2d_t input, write_only image2d_t output, constant float* mask, const int radius)
{
    sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    const int halfRadius = (int)floor(radius / 2.0);
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    float4 blurred = (float4)0.0;

    int index = 0; 
    
    for (int i = -halfRadius; i <= halfRadius; ++i)
    {
	    for (int j = -halfRadius; j <= halfRadius; ++j)
	    {				
            const int2 location = (int2)(x + j, y + i);
		    blurred += read_imagef(input, sampler, location) * mask[index];
            index++;
	    }
    }
    
    const float4 colour = read_imagef(input, sampler, (int2)(x, y));

    const float alpha = 1.5, beta = -0.5, gamma = 0.0;
    const float r = colour.x * alpha + blurred.x * beta + gamma;
    const float g = colour.y * alpha + blurred.y * beta + gamma;
    const float b = colour.z * alpha + blurred.z * beta + gamma;

    write_imagef(output, (int2)(x, y), (float4)(r, g, b, colour.w));
}

