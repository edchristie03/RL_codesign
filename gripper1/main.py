import pygame
import pymunk


from objects import Ball, Poly, Floor
from grippers import Gripper
import objects, grippers

def game(space, object):

    floor = Floor(space, 20)
    gripper = Gripper(space)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        # Function to get action from observation

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    gripper.arm.body.velocity = (-200, 0)
                if event.key == pygame.K_RIGHT:
                    gripper.arm.body.velocity = (200, 0)
                if event.key == pygame.K_UP:
                    gripper.arm.body.velocity = (0, 200)
                if event.key == pygame.K_DOWN:
                    gripper.arm.body.velocity = (0, -200)
                if event.key == pygame.K_o:
                    gripper.left_finger.body.angle -= 0.1 if gripper.left_finger.body.angle > -0.5 else 0.0
                    gripper.right_finger.body.angle += 0.1 if gripper.left_finger.body.angle > -0.5 else 0.0
                if event.key == pygame.K_c:
                    gripper.left_finger.body.angle += 0.1 if gripper.left_finger.body.angle < 1 else 0.0
                    gripper.right_finger.body.angle -= 0.1 if gripper.left_finger.body.angle < 1 else 0.0

        # White background
        display.fill((255, 255, 255))

        # Draw the objects
        # ball.draw()
        # poly.draw()
        object.draw()
        floor.draw()
        gripper.draw()

        pygame.display.update()
        clock.tick(FPS)
        space.step(1/FPS)
        gripper.arm.body.velocity = (0, 0)


if __name__ == "__main__":

    vertices = [[], [(-30, -30), (30, -30), (30, 30), (-30, 30)] ,[(-30, -30), (30, -30), (30, 30)], [(-30, -30), (30, -30), (0, 30)]]

    for vertex in vertices:

        pygame.init()
        display = pygame.display.set_mode((800, 800))
        objects.display = display
        grippers.display = display
        clock = pygame.time.Clock()
        space = pymunk.Space()
        space.gravity = (0, -1000)  # gravity
        FPS = 60

        if vertex:
            object = Poly(space, vertex)
        else:
            object = Ball(space, 30)

        game(space, object)
        pygame.quit()







