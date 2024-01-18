import pygame
import sys

pygame.init()

# Set up display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Visualization")

# Set up chat box parameters
chat_box_width = 500
chat_box_height = 100
chat_box_x = (width - chat_box_width) // 2
chat_box_y = 420
chat_font = pygame.font.Font(None, 36)
chat_text_color = (0, 0, 0)
user_input = ""

# Set up colors
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue

# Set up rectangle parameters
rectangles_width = 40
rectangles_height = 150
rectangles_x = 50  # Initial x-coordinate of rectangles

# Load icons
outagent_icon = pygame.image.load("static/Dr_Zomboss_portrait.webp")  # Path to actual image
outagent_icon = pygame.transform.scale(outagent_icon, (100, 100))  # Adjust the size if needed
inagent_icon = pygame.image.load("static/zombie.webp")  # Path to actual image
inagent_icon = pygame.transform.scale(inagent_icon, (100, 100))  # Adjust the size if needed
lawn_icon = pygame.image.load("static/lawn.webp")  # Path to actual image
law_icon = pygame.transform.scale(lawn_icon, (400, 400))  # Adjust the size if needed
phone_icon = pygame.image.load("static/telephone.jpeg")  # Path to actual image
phone_icon = pygame.transform.scale(phone_icon, (50, 50))  # Adjust the size if needed


# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                user_input = ""  # Clear the user input when Enter is pressed
            elif event.key == pygame.K_BACKSPACE:
                user_input = user_input[:-1]  # Remove the last character when Backspace is pressed
            else:
                user_input += event.unicode  # Add the pressed character to the user input

    # Draw the opened box in the middle
    screen.fill((255, 255, 255))  # Fill the screen with white
    pygame.draw.rect(screen, (0, 0, 0), (20, 200, 200, 200), 2)  # Draw the opened box


    # Draw three thin rectangles with user-specified colors
    rectangle_colors = [(255, 255, 255)] * 3  # Default to white
    user_input_rectangles = input("Enter a tuple (0, 1, 2) to set rectangle colors: ")
    try:
        colors_tuple = eval(user_input_rectangles)
        if isinstance(colors_tuple, tuple) and len(colors_tuple) == 3:
            rectangle_colors = [colors[i] for i in colors_tuple]
    except (SyntaxError, NameError):
        print("Invalid input. Default to white.")
        pass

    for i, color in enumerate(rectangle_colors):
        pygame.draw.rect(screen, color, (rectangles_x + i * 50, 250, rectangles_width, rectangles_height))

    # Draw the agent icon
    screen.blit(inagent_icon, (rectangles_x, 420))
    screen.blit(outagent_icon, (rectangles_x + 100 + chat_box_width, 420))
    
    # Draw the lawn icon
    screen.blit(law_icon, (300, 20))
    
    # Draw the phone icon
    screen.blit(phone_icon, (rectangles_x+ 100 + chat_box_width + 50, 370))

    # Draw the chat box
    user_input_chat = input("Enter a message: ")
    pygame.draw.rect(screen, (200, 200, 200), (chat_box_x, chat_box_y, chat_box_width, chat_box_height))
    chat_surface = chat_font.render(user_input_chat, True, chat_text_color)
    screen.blit(chat_surface, (chat_box_x + 10, chat_box_y + 10))

    pygame.display.flip()  # Update the display
