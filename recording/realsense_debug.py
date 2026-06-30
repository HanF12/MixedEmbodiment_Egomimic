import pyrealsense2 as rs

def list_realsense_devices():
    ctx = rs.context()
    devices = ctx.query_devices()

    if not devices:
        print("No RealSense devices found.")
        return

    print(f"Found {len(devices)} RealSense device(s):")
    for i, device in enumerate(devices):
        try:
            serial_number = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            product_line = device.get_info(rs.camera_info.product_line)
            print(f"  Device {i+1}:")
            print(f"    Name: {name}")
            print(f"    Serial Number: {serial_number}")
            print(f"    Product Line: {product_line}")
            print(f"    Firmware Version: {device.get_info(rs.camera_info.firmware_version)}")
        except Exception as e:
            print(f"  Error getting info for device {i+1}: {e}")

if __name__ == "__main__":
    list_realsense_devices()