from nerfbaselines import register

register({
    "method_class": "gsrt_method:GSRTMethod",
    "conda": {
        "environment_name": "ns",
    },
    "id": "gsrt",
    "metadata": {},
})