licenses(["restricted"])

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
	deps = [":jetson-utils-arm", ":jetson-inference-arm"],
	visibility = ["//visibility:public"],
)
