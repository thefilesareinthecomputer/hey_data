import qrcode
import os

# Define the path of the current script
script_directory = os.path.dirname(__file__)

# Define the linked URL for the QR code
url = 'https://app.getmeez.com/public/recipes/284583'

# Generate QR code
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr.add_data(url)
qr.make(fit=True)

# Create an image from the QR Code instance
img = qr.make_image(fill_color="black", back_color="grey")

# Save the image
img.save(f"{script_directory}/_sando_qr.png")
