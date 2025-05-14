import pygame
import pymunk
import numpy as np


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
                if event.key == pygame.K_x:
                    gripper.left_finger.body.angle -= 0.1 if gripper.left_finger.body.angle > -0.5 else 0.0
                if event.key == pygame.K_c:
                    gripper.left_finger.body.angle += 0.1 if gripper.left_finger.body.angle < 1 else 0.0
                if event.key == pygame.K_o:
                    gripper.right_finger.body.angle += 0.1 if gripper.left_finger.body.angle > -0.5 else 0.0
                if event.key == pygame.K_p:
                    gripper.right_finger.body.angle -= 0.1 if gripper.left_finger.body.angle < 1 else 0.0



            # Reward if left fingertip distance within threshold
            l_tip_dist = np.linalg.norm(gripper.left_finger.body.local_to_world(gripper.left_finger.shape.b) - object.body.position)
            r_tip_dist = np.linalg.norm(gripper.right_finger.body.local_to_world(gripper.right_finger.shape.b) - object.body.position)

            if l_tip_dist < 30:
                r2 = 10
            else:
                r2 = 10 - 10 * np.tanh((l_tip_dist - 30) / 100)

            if r_tip_dist < 30:
                r3 = 1
            else:
                r3 = 10 - 10 * np.tanh((r_tip_dist - 30) / 100)

            reward = r2 + r3

            print('Reward:', reward, 'r2', r2, 'r3', r3)


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

    vertices = [[], [(-30, -30), (30, -30), (30, 30), (-30, 30)] ,[(-30, -30), (30, -30), (30, 30)], [(-30, -30), (30, -30), (0, 30)],
                [(-30, -30), (30, -30), (0, 30), (-30, 30)], [(-10, -30), (0, -30), (0, 30), (-10, 30)], [(-80, -30), (80, -30), (80, 0), (-80, 0)]]

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







