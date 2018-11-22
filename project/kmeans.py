#!/usr/bin/env python
"""
K-means clustering implementation.

Notes
-----
As the below Wikipedia article points out, there is no guarantee that this implementation finds the optimal clustering result.

Reference
---------
https://en.wikipedia.org/wiki/K-means_clustering

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""
import os
import logging

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging


class NoConvergenceException(Exception):
    """Exception for not reaching convergence within specified number of iterations"""
    pass


def random_unique_choice(input_size, output_size):
    """
    Returns a subset of elements of an array consisting of elements that range from 0 to input_size-1.
    For example, if input_size is 4 and output_size is 2, two unique numbers will be picked from [0, 1, 2, 3].

    Parameters
    -----------
    input_size: int
        Size of elements from which the the subset of elements to choose from
    output_size: int
        Number of unique elements to be returned

    Returns
    -------
    out: ndarray
        Array of unique indices

    Notes
    -----
        This function is provided to ensure that the result will not contain any duplicated numbers.
        np.random.choice(a, size) may return duplicates by design.  For example, np.random.choice(4, 2)
        can return array([2, 2]).
    """

    output_array = list()
    current_array_size = input_size
    current_array = np.arange(0, input_size)

    for i in range(output_size):
        index_picked = int(np.random.choice(current_array_size, 1))
        number_picked = current_array[index_picked]
        output_array.append(number_picked)
        tmp = np.delete(current_array, index_picked)
        current_array = tmp.copy()
        current_array_size -= 1

    return np.array(output_array)


class KMeans():
    @staticmethod
    def clustering(observations, K=2, iteration_limit=1000000):
        """

        Parameters
        ----------
        observations: ndarray
            Observations
        K: int
            Number of clusters

        Returns
        -------
        observation_assignment_to_centroids: ndarray
            Array of 0-based indices of cluster assignment for each observation.  For example, [3 1 0 2] means
            the first observation is assigned to the 4th cluster (index 3).

        Raises
        ------
        ValueError
            If len(observations.shape) != 2.

        NoConvergenceException(Exception)
            If convergence is not reached within the specified number of iterations.

        Notes
        -----
        Observation has to have exactly two axes (i.e. length of the shape):
        For example,
        np.array([1, 2, 3]) will throw an exception as the shape is (3,) and the length of the shape is 1.
        2-D coordinates and 3-D coordinates work.
        [[1, 2], [3, 4]] has the shape (2, 2) and length of the shape is 2.
        [[1, 2, 3], [3, 4, 5]]) has the shape (2, 3) and length of the shape is 2.
        """

        shape = observations.shape  # e.g. (m, 2), (m, 3)
        shape_len = len(shape)

        if shape_len != 2:
            raise ValueError("Observation has to have exactly two axes such as [[1, 2, 3], [3, 4, 5]]")

        num_observations = shape[0]  # m

        # Initialize centroid positions
        #   Pick K observations randomly and assign them to centroid positions.
        #   This is called Forgy method.
        indices_picked = random_unique_choice(num_observations, K)
        log.debug("indices_picked: %s" % (str(indices_picked)))

        few_observation_positions = observations[indices_picked]
        log.debug("few_observation_positions:\n%s" % (str(few_observation_positions)))

        centroid_positions = few_observation_positions.copy()

        first_run = True
        convergence = False
        for iter in range(iteration_limit):
            if convergence:
                break

            log.debug("Centroid positions:\n%s" % (str(centroid_positions)))

            # centroid to point distance
            # For each observation, multiple columns are assigned.
            # Each column holds the distance to each centroid from the observation on the row.
            d_table = np.zeros((num_observations, K))  # distance_between_centroid_to_observation_table

            # Calculate the distance from each observation to all centeroids.
            for j in range(K):
                v = observations - centroid_positions[j]  # vector from centroid to each observation
                d = np.linalg.norm(v, axis=1)  # distance between the centroid[j] and each observation
                # Note: You need to leave 'd' as the 1-D vector.  The following will fail:
                # tmp = d.reshape(1, (d.shape[0])) # Make it a row vector
                # distance = np.transpose(tmp) # Transpose to a column vector
                d_table[:, j] = d  # Fill the column with the distance

                log.debug("d_table:\n%s" % (str(d_table)))

            # for each row (observation), find the minimum value across columns, index of which is the closest centroid.
            observation_assignment_to_centroids = np.argmin(d_table, axis=1)
            log.debug("observation_assignment_to_centroids:\n%s" % (str(observation_assignment_to_centroids)))

            # for each set of observations belonging to the same cluster, calculate the average position
            # We want to isolate a set of observation that belongs to cluster[k] for each iteration, and
            # calculate the average position.
            for j in range(K):
                indices_belonging_to_the_cluster = (
                            observation_assignment_to_centroids == j)  # True means that the element belongs to the cluster
                mean_position = np.mean(observations[indices_belonging_to_the_cluster], axis=0)
                centroid_positions[j] = mean_position

            if first_run:
                first_run = False
            else:  # If observations are assigned to the same cluster as the last run, we consider that as convergence
                if np.array_equal(observation_assignment_to_centroids, prev_observation_assignment_to_centroids):
                    log.info("Converged. Iteration: %d" % (iter + 1))
                    convergence = True
                    break

            prev_observation_assignment_to_centroids = observation_assignment_to_centroids.copy()

        if convergence is False:
            raise NoConvergenceException("Cluster assignment did not converge.")

        return observation_assignment_to_centroids
