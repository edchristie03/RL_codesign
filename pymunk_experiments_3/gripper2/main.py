import pygame
import pymunk
import numpy as np


from pymunk_experiments_3.objects import Ball, Poly, Floor, Walls
from pymunk_experiments_3.grippers import Gripper
from pymunk_experiments_3 import objects, grippers

def game(space, object):

    floor = Floor(space, 20)
    gripper = Gripper(space)
    walls = Walls(space)

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
                # Left finger 1
                if event.key == pygame.K_q:  # open
                    gripper.left_finger1.body.angle -= 0.1 if gripper.left_finger1.body.angle > -1 else 0.0
                if event.key == pygame.K_a:  # close
                    gripper.left_finger1.body.angle += 0.1 if gripper.left_finger1.body.angle < 0.7 else 0.0
                # Right finger 1
                if event.key == pygame.K_w:  # open
                    gripper.right_finger1.body.angle += 0.1 if gripper.right_finger1.body.angle < 1 else 0.0
                if event.key == pygame.K_s:  # close
                    gripper.right_finger1.body.angle -= 0.1 if gripper.right_finger1.body.angle > -0.7 else 0.0
                # Left finger 2
                if event.key == pygame.K_e:  # open
                    gripper.left_finger2.body.angle -= 0.1 if gripper.left_finger2.body.angle > -0.5 else 0.0
                if event.key == pygame.K_d:  # close
                    gripper.left_finger2.body.angle += 0.1 if gripper.left_finger2.body.angle < 1.5 else 0.0
                # Right finger 2
                if event.key == pygame.K_r:  # open
                    gripper.right_finger2.body.angle += 0.1 if gripper.right_finger2.body.angle < 0.5 else 0.0
                if event.key == pygame.K_f:  # close
                    gripper.right_finger2.body.angle -= 0.1 if gripper.right_finger2.body.angle > -1.5 else 0.0


        # White background
        display.fill((255, 255, 255))

        # Draw the objects
        object.draw()
        floor.draw()
        gripper.draw()
        walls.draw()

        pygame.display.update()
        clock.tick(FPS)
        space.step(1/FPS)
        gripper.arm.body.velocity = (0, 0)


if __name__ == "__main__":

    shapes = {
        "circle": []
    }


    for name, vertex in shapes.items():

        pygame.init()
        display = pygame.display.set_mode((800, 800))
        objects.display = display
        grippers.display = display
        clock = pygame.time.Clock()
        space = pymunk.Space()
        space.gravity = (0, -1000)  # gravity
        FPS = 200

        if vertex:
            object = Poly(space, vertex)
        else:
            object = Ball(space, 30)

        game(space, object)
        pygame.quit()







