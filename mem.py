def mem(N):
    print(N,"----------------")
    xyz = 3
    scale = 3
    rotation = 4
    opacity = 1
    sh = 16*3

    nparams = N*(xyz+scale+rotation+opacity+sh)

    params_size = nparams*4
    grad_size = params_size

    exp_avg = grad_size
    exp_avg_sq = params_size
    adam_max_size = exp_avg + exp_avg_sq

    act_size = N*(scale+rotation+opacity*2)*4

    densif_size = N*2*4

    mesh_size = (N*3*12+N*3*20)*4
    gas_size = N*1024

    total_scene = params_size + grad_size + adam_max_size + act_size + densif_size + mesh_size + gas_size
    print("scene mem Kb",total_scene/1024)

    npix = 876*584
    col_buf = npix*3*4
    trans_buf = npix*4
    dist_buf = npix*4
    dL_dC = npix*3*4
    gt = npix*3*4
    rays = npix*3*4
    sort_buf = 512*npix*12

    total_im = col_buf + trans_buf + dist_buf + dL_dC + gt + rays + sort_buf
    print("im mem Kb",total_im/1024)

    total = total_scene+total_im
    print(total/1024**3, "Gb")

    print("----------------")

#mem(700000) #-> actual ~5e9
mem(273731) #-> actual ~3e9

