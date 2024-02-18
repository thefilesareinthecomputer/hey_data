from PIL import Image, ImageDraw, ImageFont
import os
import qrcode

def generate_qr_with_logo_text(data, logo_path, text, qr_color='#777777', bg_color='#4A4A4A'):
    # Path to the Open Sans Regular font
    font_path = os.path.join(os.path.dirname(__file__), 'Open_Sans', 'static', 'OpenSans-Regular.ttf')

    # Generate QR code
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_H
    )
    qr.add_data(data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color=qr_color, back_color=bg_color).convert('RGB')
    
    # Add logo to QR code
    logo = Image.open(logo_path)
    logo_size = qr_img.size[0] // 5
    logo = logo.resize((logo_size, logo_size))
    pos = ((qr_img.size[0] - logo.size[0]) // 2, (qr_img.size[1] - logo.size[1]) // 2)
    qr_img.paste(logo, pos)

    # Create a drawing context
    draw = ImageDraw.Draw(qr_img)

    # Specify the font size and type
    font_size = 32
    font = ImageFont.truetype(font_path, font_size)

    # Calculate text dimensions
    text_width, text_height = draw.textsize(text, font)

    # Extend QR code image to add text at the bottom
    extended_img_height = qr_img.size[1] + text_height + 20  # 20 is padding
    extended_img = Image.new('RGB', (qr_img.size[0], extended_img_height), bg_color)
    extended_img.paste(qr_img, (0, 0))

    # Calculate text position
    text_x = (extended_img.size[0] - text_width) // 2
    text_y = qr_img.size[1] + 10  # 10 is padding from the bottom of the QR code

    # Create a new drawing context
    draw = ImageDraw.Draw(extended_img)

    # Draw text
    draw.text((text_x, text_y), text, fill=qr_color, font=font)

    return extended_img

# Usage
data = 'https://app.getmeez.com/public/recipes/284583'
script_directory = os.path.dirname(__file__)
logo_path = os.path.join(script_directory, '_logo.png')
text = 'Wagyu Ribeye Sandwich'
qr_img = generate_qr_with_logo_text(data, logo_path, text)
output_path = os.path.join(script_directory, '_qr_logo_text.png')
qr_img.save(output_path)
qr_img.show()














































# from PIL import Image, ImageDraw, ImageFont
# import os
# import qrcode

# def generate_qr_with_logo_text(data, 
#                                logo_path, 
#                                text, 
#                                qr_color='#777777', 
#                                bg_color='#4A4A4A',
#                                text_width=150, 
#                                text_height=40, 
#                                text_padding=0):
#     # Path to the Open Sans Regular font
#     font_path = os.path.join(os.path.dirname(__file__), 'Open_Sans', 'static', 'OpenSans-Regular.ttf')

#     # Generate QR code
#     qr = qrcode.QRCode(
#         error_correction=qrcode.constants.ERROR_CORRECT_H
#     )
#     qr.add_data(data)
#     qr.make()
#     qr_img = qr.make_image(fill_color=qr_color, back_color=bg_color).convert('RGB')
    
#     # Add logo to QR code
#     logo = Image.open(logo_path)
#     logo_size = qr_img.size[0] // 5  # Adjust logo size here if needed
#     logo = logo.resize((logo_size, logo_size))
#     pos = ((qr_img.size[0] - logo.size[0]) // 2, (qr_img.size[1] - logo.size[1]) // 2)
#     qr_img.paste(logo, pos)

#     # Extend QR code image to add text at the bottom with manual padding
#     extended_img_height = qr_img.size[1] + text_height + text_padding
#     extended_img = Image.new('RGB', (qr_img.size[0], extended_img_height), bg_color)
#     extended_img.paste(qr_img, (0, 0))

#     # Add text
#     draw = ImageDraw.Draw(extended_img)
#     font_size = 32  # Adjust font size here if needed
#     font = ImageFont.truetype(font_path, font_size)
#     text_x = (qr_img.size[0] - text_width) // 2  # Center the text manually
#     text_y = qr_img.size[1] + text_padding  # Position text manually

#     # Draw text at the manually calculated position
#     draw.text((text_x, text_y), text, fill=qr_color, font=font)

#     return extended_img

# # Usage
# data = 'https://app.getmeez.com/public/recipes/284583'
# script_directory = os.path.dirname(__file__)
# logo_path = os.path.join(script_directory, '_logo.png')
# text = 'Wagyu Ribeye Sandwich'
# qr_img = generate_qr_with_logo_text(data, 
#                                     logo_path, 
#                                     text, 
#                                     text_width=150, 
#                                     text_height=40, 
#                                     text_padding=0)
# output_path = os.path.join(script_directory, '_qr_logo_text.png')
# qr_img.save(output_path)
# qr_img.show()



















































# from PIL import Image, ImageDraw, ImageFont
# import os
# import qrcode

# def generate_qr_with_logo_text(data, logo_path, text, qr_color='#777777', bg_color='#4A4A4A'):
#     # Path to the Open Sans Regular font
#     font_path = os.path.join(os.path.dirname(__file__), 'Open_Sans', 'static', 'OpenSans-Regular.ttf')

#     # Generate QR code
#     qr = qrcode.QRCode(
#         error_correction=qrcode.constants.ERROR_CORRECT_H
#     )
#     qr.add_data(data)
#     qr.make()
#     qr_img = qr.make_image(fill_color=qr_color, back_color=bg_color).convert('RGB')
    
#     # Add logo to QR code
#     logo = Image.open(logo_path)
#     logo_size = qr_img.size[0] // 5
#     logo = logo.resize((logo_size, logo_size))
#     pos = ((qr_img.size[0] - logo.size[0]) // 2, (qr_img.size[1] - logo.size[1]) // 2)
#     qr_img.paste(logo, pos)

#     # Extend QR code image to add text at the bottom
#     text_area_height = 20
#     extended_img = Image.new('RGB', (qr_img.size[0], qr_img.size[1] + text_area_height), bg_color)
#     extended_img.paste(qr_img, (0, 0))

#     # Add text
#     draw = ImageDraw.Draw(extended_img)
#     font_size = 32
#     font = ImageFont.truetype(font_path, font_size)

#     # Use getbbox to get the text size
#     bbox = draw.textbbox((0, 0), text, font=font)
#     text_width = bbox[2] - bbox[0]
#     text_height = bbox[3] - bbox[1]
#     text_x = (extended_img.size[0] - text_width) // 2
#     text_y = qr_img.size[1] + (text_area_height - text_height) // 2

#     # Draw text at the calculated position
#     draw.text((text_x, text_y), text, fill=qr_color, font=font)

#     return extended_img

# # Usage
# data = 'https://app.getmeez.com/public/recipes/284583'
# script_directory = os.path.dirname(__file__)
# logo_path = os.path.join(script_directory, '_logo.png')
# text = 'Wagyu Ribeye Sandwich'
# qr_img = generate_qr_with_logo_text(data, logo_path, text)
# output_path = os.path.join(script_directory, '_qr_logo_text.png')
# qr_img.save(output_path)
# qr_img.show()
