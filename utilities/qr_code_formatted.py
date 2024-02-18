from PIL import Image
import os
import qrcode

def generate_qr_with_logo(data, logo_path, qr_color='#777777', bg_color='#3E3E3E'):
    # Generate QR code
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H
    )
    qr.add_data(data)
    qr.make()
    qr_img = qr.make_image(fill_color=qr_color, back_color=bg_color).convert('RGB')
    
    # Add logo to QR code
    logo = Image.open(logo_path)
    logo_size = qr_img.size[0] // 5  # Logo size is 1/5 of QR code size
    logo = logo.resize((logo_size, logo_size))
    pos = ((qr_img.size[0] - logo.size[0]) // 2, (qr_img.size[1] - logo.size[1]) // 2)
    qr_img.paste(logo, pos, logo)

    return qr_img

# Usage
data = 'https://app.getmeez.com/public/recipes/284583'
script_directory = os.path.dirname(__file__)
logo_path = f'{script_directory}/_logo.png'
qr_img = generate_qr_with_logo(data, logo_path)
qr_img.save(f'{script_directory}/_qr_logo.png')  # Save the QR code to a file
qr_img.show()  # This will display the image in the default image viewer




































# from PIL import Image, ImageDraw, ImageFont
# import os
# import qrcode

# def generate_qr_with_logo_text(data, logo_path, text, qr_color='#777777', bg_color='#4A4A4A'):
#     # Generate QR code
#     qr = qrcode.QRCode(
#         error_correction=qrcode.constants.ERROR_CORRECT_H
#     )
#     qr.add_data(data)
#     qr.make()
#     qr_img = qr.make_image(fill_color=qr_color, back_color=bg_color).convert('RGB')
    
#     # Add logo to QR code
#     logo = Image.open(logo_path)
#     logo_size = qr_img.size[0] // 5  # Logo size is 1/5 of QR code size
#     logo = logo.resize((logo_size, logo_size))
#     pos = ((qr_img.size[0] - logo.size[0]) // 2, (qr_img.size[1] - logo.size[1]) // 2)
#     qr_img.paste(logo, pos, logo)

#     # Extend QR code image to add text at the bottom
#     extended_img = Image.new('RGB', (qr_img.size[0], qr_img.size[1] + 50), bg_color)
#     extended_img.paste(qr_img, (0, 0))

#     # Add text
#     draw = ImageDraw.Draw(extended_img)
#     font = ImageFont.load_default()  # Default font, adjust size if needed
    
#     # Use getbbox for text size calculation
#     bbox = draw.textbbox((0, 0), text, font=font)
#     text_width = bbox[2] - bbox[0]
#     text_height = bbox[3] - bbox[1]
    
#     text_pos = ((extended_img.size[0] - text_width) // 2, qr_img.size[1] + (50 - text_height) // 2)  # Adjust text position
#     draw.text(text_pos, text, fill=qr_color, font=font)

#     return extended_img

# # Usage
# data = 'https://app.getmeez.com/public/recipes/284583'
# script_directory = os.path.dirname(__file__)
# logo_path = f'{script_directory}/_logo.png'
# text = 'Wagyu Ribeye Sandwich'
# qr_img = generate_qr_with_logo_text(data, logo_path, text)
# qr_img.save(f'{script_directory}/_qr_logo_text.png')  # Save the QR code to a file
# qr_img.show()  # This will display the image in the default image viewer

