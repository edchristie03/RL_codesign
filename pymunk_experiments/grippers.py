import pygame
import pymunk

def convert_coordinates(point):
    return int(point.x), 800 - int(point.y)

class Gripper():
    def __init__(self, space, base_width=200):

        # Create the base of the gripper
        self.base = Base(space, 250, base_width)
        # Create the arm
        self.arm = Arm(space, self.base)
        # Create the left finger
        self.left_finger = Finger1(space, self.base.body.local_to_world(self.base.shape.a), self.base, side='left')
        # Create the right finger
        self.right_finger = Finger1(space, self.base.body.local_to_world(self.base.shape.b), self.base, side='right')

    def draw(self):
        # Draw
        self.arm.draw()
        self.base.draw()
        self.left_finger.draw()
        self.right_finger.draw()

class Gripper2():
    def __init__(self, space, design_vector=(200, 120, 120, 120, 120)):


        base_width = design_vector[0]
        lf1_length = design_vector[1]
        rf1_length = design_vector[2]
        lf2_length = design_vector[3]
        rf2_length = design_vector[4]

        height = max(lf1_length + lf2_length, rf1_length + rf2_length) + 150

        # Create the base of the gripper
        self.base = Base(space, height, base_width)
        # Create the arm
        self.arm = Arm(space, self.base)
        # Create the left finger part 1
        self.left_finger1 = Finger1(space, self.base.body.local_to_world(self.base.shape.a), self.base, lf1_length, side='left')
        # Create the left finger part 2
        self.left_finger2 = Finger2(space, self.left_finger1.body.local_to_world(self.left_finger1.shape.b), lf2_length, self.left_finger1)
        # Create the right finger part 1
        self.right_finger1 = Finger1(space, self.base.body.local_to_world(self.base.shape.b), self.base, rf1_length, side='right')
        # Create the right finger part 2
        self.right_finger2 = Finger2(space, self.right_finger1.body.local_to_world(self.right_finger1.shape.b), rf2_length, self.right_finger1)

    def draw(self):
        # Draw
        self.arm.draw()
        self.base.draw()
        self.left_finger1.draw()
        self.left_finger2.draw()
        self.right_finger1.draw()
        self.right_finger2.draw()


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
        self.body.position = (400, height)
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
    def __init__(self, space, anchor, base, length=120, side='left'):
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = anchor
        self.shape = pymunk.Segment(self.body, (0, 0), (0, -length), 5)
        self.shape.density = 1
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
        self.motor.max_force = 5e20
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
        self.shape.density = 1
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



