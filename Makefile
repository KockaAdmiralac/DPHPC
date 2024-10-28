format:
	find . -iname '*.h' -o -iname '*.c' -o -iname '*.cuh' -o -iname '*.cu' | xargs clang-format -i
	black .