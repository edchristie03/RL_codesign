import pygame
import pymunk

def convert_coordinates(point):
    return int(point.x), 800 - int(point.y)

class Gripper():
    def __init__(self, space, design_vector):

        length = design_vector[0]
        number = design_vector[1]

        # Create the base of the gripper
        self.base = Base(space, 400, 100)
        # Create the arm
        self.arm = Arm(space, self.base)

        # Store finger segments in lists
        self.right_fingers = []

        # Create right finger chain
        for i in range(number):
            if i == 0:
                # First finger connects to base
                connection_point = self.base.body.local_to_world(self.base.shape.b)
                parent_body = self.base
                finger = Finger1(space, connection_point, parent_body, length, side='right')
            else:
                # Subsequent fingers connect to previous finger
                connection_point = self.right_fingers[i - 1].body.local_to_world(self.right_fingers[i - 1].shape.b)
                parent_body = self.right_fingers[i - 1]
                finger = Finger2(space, connection_point, length, parent_body)

            # Create finger segment

            self.right_fingers.append(finger)


    def draw(self):
        # Draw
        self.arm.draw()
        self.base.draw()
        for finger in self.right_fingers:
            finger.draw()

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
    def __init__(self, space, height, base_width):
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = (200, height)
        self.shape = pymunk.Segment(self.body, (-base_width/2, 0), (base_width/2, 0), 5)
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

class Finger1():
    def __init__(self, space, anchor, base, length=120, side='left', motor_force=5e20):
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = anchor
        self.shape = pymunk.Segment(self.body, (0, 0), (0, -length), 5)
        self.shape.density = 0.2
        self.shape.friction = 0.5
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
        self.motor.max_force = motor_force
        space.add(self.motor)

    def draw(self):
        world_a = self.body.local_to_world(self.shape.a)
        world_b = self.body.local_to_world(self.shape.b)
        x1, y1 = convert_coordinates(world_a)
        x2, y2 = convert_coordinates(world_b)
        # Draw the line
        pygame.draw.line(display, (0, 255, 0), (x1, y1), (x2, y2), int(self.shape.radius * 2))

class Finger2():
    def __init__(self, space, anchor, length, finger_above):
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = anchor
        self.shape = pymunk.Segment(self.body, (0, 0), (0, -length), 5)
        self.shape.density = 0.2
        self.shape.friction = 0.5
        space.add(self.body, self.shape)

        # Create the joint
        self.joint = pymunk.PivotJoint(finger_above.body, self.body, finger_above.body.local_to_world(finger_above.shape.b))
        self.joint.collide_bodies = False
        space.add(self.joint)

        # Create limit
        # self.limit = pymunk.RotaryLimitJoint(finger_above.body, self.body, -1, 1)
        # space.add(self.limit)

        # Create motor
        self.motor = pymunk.SimpleMotor(finger_above.body, self.body, 0)
        self.motor.max_force = 5e20
        space.add(self.motor)

    def draw(self):
        world_a = self.body.local_to_world(self.shape.a)
        world_b = self.body.local_to_world(self.shape.b)
        x1, y1 = convert_coordinates(world_a)
        x2, y2 = convert_coordinates(world_b)
        # Draw the line
        pygame.draw.line(display, (0, 255, 0), (x1, y1), (x2, y2), int(self.shape.radius * 2))



