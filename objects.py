import pygame
import pymunk


def convert_coordinates(point):
    return int(point.x), 800 - int(point.y)

class Ball():
    def __init__(self, space, radius):
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)      # point like object
        self.body.position = (400, 100)
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.density = 5
        self.shape.elasticity = 0.2
        self.shape.friction = 1.5
        self.body.angular_damping = 0.1
        space.add(self.body, self.shape)

    def draw(self):
        x, y = convert_coordinates(self.body.position)
        pygame.draw.circle(display, (0, 0, 255), (int(x), int(y)), int(self.shape.radius))

class Poly():
    def __init__(self, space, vertices):
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = (400, 100)
        self.shape = pymunk.Poly(self.body, vertices, radius=0)
        self.shape.density = 5
        self.shape.elasticity = 0.2
        self.shape.friction = 0.5
        space.add(self.body, self.shape)

    def draw(self):
        # Transform local vertices into world-space
        verts_world = [v.rotated(self.body.angle) + self.body.position for v in self.shape.get_vertices()]
        # Then convert to Pygame coordinates
        verts_screen = [convert_coordinates(v) for v in verts_world]
        pygame.draw.polygon(display, (0, 0, 255), verts_screen)

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

class Walls():
    def __init__(self, space):
        self.wall1 = Wall(space, 'left')
        self.wall2 = Wall(space, 'right')

    def draw(self):
        self.wall1.draw()
        self.wall2.draw()

class Wall():
    def __init__(self, space, side):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        if side == 'left':
            self.shape = pymunk.Segment(self.body, (0, 0), (0, 800), 1)
        else:
            self.shape = pymunk.Segment(self.body, (800, 0), (800, 800), 1)
        self.shape.elasticity = 0.5
        self.shape.friction = 0.7
        space.add(self.body, self.shape)


    def draw(self):
        x1, y1 = convert_coordinates(self.shape.a)
        x2, y2 = convert_coordinates(self.shape.b)
        pygame.draw.line(display, (0, 0, 0), (x1, y1), (x2, y2), int(self.shape.radius * 2))


