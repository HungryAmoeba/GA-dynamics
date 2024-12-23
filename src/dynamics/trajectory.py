import numpy as np

class GraphDynamics:
    def __init__(self, model, data, pos, adjacency):
        self.model = model
        self.data = data
        self.pos = pos
        self.adjacency = adjacency
        self.time_elapsed = 0
    
    # TODO: see if its better to make a class to wrap the model and data and data trajectory

class TrajectoryResampler:
    def __init__(self, reference_frames, framerate, total_time):
        self.reference_frames = reference_frames
        self.framerate = framerate
        self.total_time = total_time
        self.num_frames = len(reference_frames)
        self.target_num_frames = int(total_time * framerate)
        self.points_between_frames = self.target_num_frames // self.num_frames

    def resample_trajectory(self, reverse = False):
        """Interpolate frames to match a target framerate and total time. 
        Linearly interpolates positions to upscale the number of frames.
        """
        num_frames = len(self.reference_frames)
        target_num_frames = int(self.total_time * self.framerate)
        points_between_frames = target_num_frames / num_frames
        num_nodes = len(self.reference_frames[0])

        # interpolate the positions
        new_positions = []
        for i in range(num_frames - 1):
            start_pos = self.reference_frames[i]
            end_pos = self.reference_frames[i + 1]
            for j in range(self.points_between_frames):
                # pos and end_pos are dictionaries of node number to position
                new_pos = np.zeros((num_nodes,3))
                for node in start_pos:
                    start = np.array(start_pos[node])
                    end = np.array(end_pos[node])
                    pos_at_node = start + (end - start) * j / self.points_between_frames
                    new_pos[node] = pos_at_node
                    #new_pos[node] = tuple([float(x) for x in pos_at_node])
                    
                new_positions.append(new_pos)
                #new_pos = start_pos + (end_pos - start_pos) * j / self.points_between_frames
                #new_positions.append(new_pos)
        if reverse:
            new_positions = new_positions[::-1]
        return np.array(new_positions)
    