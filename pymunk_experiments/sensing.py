import pymunk

class Sensors():

    def __init__(self, gripper, object):

        self.gripper = gripper
        self.object = object

    def get_three_segment_forces(self):
        """
        Alternative implementation using actual collision detection.
        More accurate but potentially more expensive computationally.
        """
        finger_shapes = [
            self.gripper.left_finger1.shape,
            self.gripper.left_finger2.shape,
            self.gripper.right_finger1.shape,
            self.gripper.right_finger2.shape
        ]

        finger_bodies = [
            self.gripper.left_finger1.body,
            self.gripper.left_finger2.body,
            self.gripper.right_finger1.body,
            self.gripper.right_finger2.body
        ]

        all_forces = []

        for shape, body in zip(finger_shapes, finger_bodies):
            # Get collision info
            collision_info = shape.shapes_collide(self.object.shape)

            if collision_info.points:
                # Classify contact points into 3 segments
                segment_forces = [0.0, 0.0, 0.0]  # [proximal, middle, distal]

                # Get finger bounds in local coordinates
                if hasattr(shape, 'a') and hasattr(shape, 'b'):
                    start_local = shape.a
                    end_local = shape.b
                    finger_length = (end_local - start_local).length
                else:
                    finger_length = shape.radius * 2
                    start_local = pymunk.Vec2d(-shape.radius, 0)

                for contact_point in collision_info.points:
                    # Transform contact point to local coordinates
                    world_contact = contact_point.point_a  # Contact point on finger
                    local_contact = body.world_to_local(world_contact)

                    # Determine which segment this contact belongs to
                    if finger_length > 0:
                        # Calculate position along finger (0 to 1)
                        progress = max(0, min(1, (local_contact - start_local).dot(
                            end_local - start_local) / finger_length ** 2))

                        # Assign to segment
                        if progress < 0.33:
                            segment_idx = 0  # Proximal
                        elif progress < 0.67:
                            segment_idx = 1  # Middle
                        else:
                            segment_idx = 2  # Distal

                        # Calculate force from penetration depth
                        penetration_force = abs(contact_point.distance) * 10.0  # Scale factor
                        segment_forces[segment_idx] += min(penetration_force, 50.0)

                all_forces.extend(segment_forces)
            else:
                all_forces.extend([0.0, 0.0, 0.0])  # No contact

        return all_forces


