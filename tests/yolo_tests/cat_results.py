from PIL import Image
import os


def create_gif(image_folder, start, end, output_path, frame_duration=500):
    frames = []

    # Loop through the range of numbers
    for i in range(start, end + 1):
        filename = f"cam5_{i:06d}.jpg"
        filepath = os.path.join(image_folder, filename)

        # Check if the file exists and open it
        if os.path.exists(filepath):
            with Image.open(filepath) as img:
                img.thumbnail((540, 320))
                # img = img.convert("P", palette=Image.ADAPTIVE, colors=128)

                frames.append(img.copy())

    # Save the frames as a gif
    if frames:
        frames[0].save(output_path, save_all=True, append_images=frames[1:], optimize=False, duration=frame_duration, loop=0)
        print(f"GIF created at {output_path}")
    else:
        print("No images were found or compiled into a GIF.")

# Usage
image_folder = 'data/debug_6000_no_bgremoval/image_data_2' # Change to your image folder path
start = 6279
end = 7663
output_gif = 'improved.gif'  # Change to your desired output path
frame_duration = 30  # Duration of each frame in milliseconds

create_gif(image_folder, start, end, output_gif, frame_duration)
