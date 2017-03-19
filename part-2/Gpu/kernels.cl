
kernel void unsharp_mask(read_only image2d_t input, write_only image2d_t output, constant float* mask, const int radius)
{
    sampler_t sampler =  CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    const float alpha = 1.5, beta = -0.5, gamma = 0.0;
    const int halfRadius = (int)floor(radius / 2.0);
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    float4 blurred = (float4)0.0;
    
    int maskIndex = 0;

    for (int i = -halfRadius; i <= halfRadius; ++i)
    {
	    for (int j = -halfRadius; j <= halfRadius; ++j)
	    {				
            const int2 location = (int2)(x + j, y + i);
		    blurred += read_imagef(input, sampler, location) * mask[maskIndex];
            maskIndex++;
	    }
    }
   
    const float4 colour = read_imagef(input, sampler, (int2)(x, y));
    const float4 result = colour * alpha + blurred * beta + gamma; 

    write_imagef(output, (int2)(x, y), result);
}

