from _nerfbaselines import register

register({
    "method_class": "fsgs_method:GaussianSplatting",
    "conda": {
        "environment_name": "figheye_gs",
    },
    "id": "fsgs",
    "metadata": {},
})