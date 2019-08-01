licenses(["restricted"])

config_setting(
	name = "arm_build",
	values = {"cpu": "aarch64"},
)

config_setting(
	name = "x86_build",
	values = {"cpu": "x86"},
)

cc_import(
	name = "jetson-inference-x86",
	shared_library = "x86_64/lib/libjetson-inference.so",
	visibility = ["//visibility:public"],
)

cc_import(
	name = "jetson-utils-x86",
	shared_library = "x86_64/lib/libjetson-utils.so",
	visibility = ["//visibility:public"],
)

cc_import(
	name = "jetson-inference-arm",
	shared_library = "libtest.so",
	visibility = ["//visibility:public"],
)

cc_import(
	name = "jetson-utils-arm",
	shared_library = "aarch64/lib/libjetson-utils.so",
	visibility = ["//visibility:public"],
)

cc_library(
	name = "jetson",
	hdrs = glob(["*.h", "utils/*.h", "utils/cuda/*.h"]),
	includes = ["utils/", "utils/cuda/"],
	data = glob(["data/**"]),
	deps = select({
			":arm_build": [":jetson-inference-arm"],
			":x86_build": [":jetson-inference-x86"],
			"//conditions:default": [":jetson-inference-x86"],
		}),
	visibility = ["//visibility:public"],
)
