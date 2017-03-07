
kernel void add(ulong n,
   global const float *a,
   global const float *b,
   global float *c)
{
    size_t i = get_global_id(0);
    if (i < n) {
       c[i] = a[i] + b[i];
    }
};

kernel void copy(read_only image2d_t input, write_only image2d_t output)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    float4 colour = read_imagef(input, sampler, (int2)(x, y));
    write_imagef(output, (int2)(x, y), colour);
}


kernel void box_blur()
{ 
    size_t x = get_global_id(0);
    size_t y = get_global_id(1); 
}

kernel void add_weighted()
{ 
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

}

kernel void gaussian_blur()
{ 
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
}

