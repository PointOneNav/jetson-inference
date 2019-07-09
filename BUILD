licenses(["restricted"])
cc_import(
	name = "jetson-inference",
	shared_library = "x86_64/lib/libjetson-inference.so",
	visibility = ["//visibility:public"],
)

cc_import(
	name = "jetson-utils",
	shared_library = "x86_64/lib/libjetson-utils.so",
	visibility = ["//visibility:public"],
)

cc_library(
	name = "jetson",
	hdrs = glob(["*.h", "utils/*.h", "utils/cuda/*.h"]),
	includes = ["utils/", "utils/cuda/"],
	data = glob(["data/**"]),
	deps = [":jetson-utils", ":jetson-inference"],
	visibility = ["//visibility:public"],
)
