from nerfbaselines import register

register({
    "method_class": "gsrt_method:GSRTMethod",
    "conda": {
        "environment_name": "gsrt_method",
        "python_version": "3.8"
    },
    "id": "gsrt-method",
    "metadata": {},
})