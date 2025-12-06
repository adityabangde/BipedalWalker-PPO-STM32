import sys
import os

def convert_tflite_to_header(tflite_path, header_path, var_name="g_model"):
    guard = os.path.basename(header_path).replace(".", "_").upper()

    with open(tflite_path, "rb") as f:
        data = f.read()

    with open(header_path, "w") as h:
        h.write(f"#ifndef {guard}\n")
        h.write(f"#define {guard}\n\n")
        h.write("// Auto-generated header containing TFLite model data\n\n")
        h.write(f"const unsigned char {var_name}[] = {{\n    ")

        for i, b in enumerate(data):
            h.write(f"0x{b:02x}")
            if i != len(data) - 1:
                h.write(", ")
            if (i + 1) % 12 == 0:
                h.write("\n    ")

        h.write("\n};\n\n")
        h.write(f"const unsigned int {var_name}_len = {len(data)};\n\n")
        h.write(f"#endif // {guard}\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 tflite_to_header.py model.tflite model.h [var_name]")
        sys.exit(1)

    tflite_path = sys.argv[1]
    header_path = sys.argv[2]
    var_name = sys.argv[3] if len(sys.argv) > 3 else "g_model"

    convert_tflite_to_header(tflite_path, header_path, var_name)
    print(f"Done. Header written to {header_path}")
