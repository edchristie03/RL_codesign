import pygame
import pymunk
import numpy as np
import math

def convert_coordinates(point):
    return int(point.x), 800 - int(point.y)

class Ball():
    def __init__(self, space, radius):
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)      # point like object
        self.body.position = (400, 100)
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.density = 0.5
        self.shape.elasticity = 0.5
        self.shape.friction = 1
        self.body.angular_damping = 0.1
        space.add(self.body, self.shape)

    def draw(self):
        x, y = convert_coordinates(self.body.position)
        pygame.draw.circle(display, (0, 0, 255), (int(x), int(y)), int(self.shape.radius))

class Poly():
    def __init__(self, space, vertices):
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = (400, 100)
        self.shape = pymunk.Poly(self.body, vertices)
        self.shape.density = 0.5
        self.shape.elasticity = 0.5
        self.shape.friction = 0.5
        space.add(self.body, self.shape)

    def draw(self):
        # Transform local vertices into world-space
        verts_world = [v.rotated(self.body.angle) + self.body.position for v in self.shape.get_vertices()]
        # Then convert to Pygame coordinates
        verts_screen = [convert_coordinates(v) for v in verts_world]
        pygame.draw.polygon(display, (0, 255, 0), verts_screen)

class Floor():
    def __init__(self, space, radius):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.shape = pymunk.Segment(self.body, (0, 50), (800, 50), radius)
        self.shape.elasticity = 0.5
        self.shape.friction = 1
        space.add(self.body, self.shape)

    def draw(self):
        x1, y1 = convert_coordinates(self.shape.a)
        x2, y2 = convert_coordinates(self.shape.b)
        pygame.draw.line(display, (0, 0, 0), (x1, y1), (x2, y2), int(self.shape.radius * 2))

class Gripper():
    def __init__(self, space):

        # Create the base of the gripper
        self.base = Base(space)
        # Create the arm
        self.arm = Arm(space, self.base)
        # Create the left finger
        self.left_finger = Finger(space, self.base.body.local_to_world(self.base.shape.a), self.base, side='left')
        # Create the right finger
        self.right_finger = Finger(space, self.base.body.local_to_world(self.base.shape.b), self.base, side='right')

    def draw(self):
        # Draw
        self.arm.draw()
        self.base.draw()
        self.left_finger.draw()
        self.right_finger.draw()

class Arm():
    def __init__(self, space, base):
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.body.position = base.body.position
        self.shape = pymunk.Segment(self.body, (0, 0), (0, 150), 5)
        space.add(self.body, self.shape)

        self.joint = pymunk.PivotJoint(self.body, base.body, base.body.position)
        self.joint.collide_bodies = False
        space.add(self.joint)

        # Create limit
        self.limit = pymunk.RotaryLimitJoint(base.body, self.body, 0, 0)
        space.add(self.limit)


    def draw(self):
        # Transform local endpoints into world-space
        world_a = self.body.local_to_world(self.shape.a)
        world_b = self.body.local_to_world(self.shape.b)
        # Then convert to Pygame coordinates
        x1, y1 = convert_coordinates(world_a)
        x2, y2 = convert_coordinates(world_b)
        # Draw the line
        pygame.draw.line(display, (0, 255, 0), (x1, y1), (x2, y2), int(self.shape.radius * 2))

class Base():
    def __init__(self, space):
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = (400, 250)
        self.shape = pymunk.Segment(self.body, (-100, 0), (100, 0), 5)
        self.shape.density = 1
        space.add(self.body, self.shape)

    def draw(self):
        # Transform local endpoints into world-space
        world_a = self.body.local_to_world(self.shape.a)
        world_b = self.body.local_to_world(self.shape.b)
        # Then convert to Pygame coordinates
        x1, y1 = convert_coordinates(world_a)
        x2, y2 = convert_coordinates(world_b)
        # Draw the line
        pygame.draw.line(display, (255, 0, 0), (x1, y1), (x2, y2), int(self.shape.radius * 2))

class Finger():
    def __init__(self, space, anchor, base, side='left'):
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = anchor
        self.shape = pymunk.Segment(self.body, (0, 0), (0, -120), 5)
        self.shape.density = 1
        space.add(self.body, self.shape)

        # Create the joint
        if side == 'left':
            self.joint = pymunk.PivotJoint(base.body, self.body, base.body.local_to_world(base.shape.a))
        elif side == 'right':
            self.joint = pymunk.PivotJoint(base.body, self.body, base.body.local_to_world(base.shape.b))

        self.joint.collide_bodies = False
        space.add(self.joint)

        # Create limit
        self.limit = pymunk.RotaryLimitJoint(base.body, self.body, -1, 1)
        space.add(self.limit)

        # Create motor
        self.motor = pymunk.SimpleMotor(base.body, self.body, 0)
        self.motor.max_force = 5e20
        space.add(self.motor)

    def draw(self):
        world_a = self.body.local_to_world(self.shape.a)
        world_b = self.body.local_to_world(self.shape.b)
        x1, y1 = convert_coordinates(world_a)
        x2, y2 = convert_coordinates(world_b)
        # Draw the line
        pygame.draw.line(display, (0, 255, 0), (x1, y1), (x2, y2), int(self.shape.radius * 2))


def game(space, object):
    # ball = Ball(space, 30)
    # poly = Poly(space, [(0, 0), (50, 0), (50, 50), (0, 50)])
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

    vertices = [[], [(-25, 0), (25, 0), (25, 50), (-25, 50)] ,[(-25, 0), (25, 0), (25, 50)], [(-25, 0), (25, 0), (0, 50)]]

    for vertex in vertices:

        pygame.init()
        display = pygame.display.set_mode((800, 800))
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





    # # Run the game
    # game(space)
    # pygame.quit()





