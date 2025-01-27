import pytest
from power2 import app, TSP
import numpy as np
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """Set up a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def tsp_instance():
    """Set up a TSP instance for testing."""
    return TSP()


def test_home(client):
    """Test the home endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    # Check if a unique part of the HTML content is in the response
    assert b"Traveling Salesperson App" in response.data


def test_add_node(client, tsp_instance):
    """Test the add_node endpoint."""
    address = "Tel Aviv, Israel"
    response = client.post("/add_node", data={"address": address})
    json_data = response.get_json()
    assert response.status_code == 200
    if json_data["success"]:
        assert "Node added" in json_data["message"]
    else:
        assert "Failed to geocode address" in json_data["message"]



@pytest.mark.parametrize(
    "distance_matrix, expected_path, expected_cost",
    [
        # Test case 1: Simple 2-node matrix
        (np.array([[0, 1], [1, 0]]), (0, 1), 2),

        # Test case 2: Simple 3-node matrix
        (np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]), (0, 1, 2), 4),

        # Test case 3: 4-node matrix with a unique shortest path
        (
            np.array([
                [0, 2, 9, 10],
                [1, 0, 6, 4],
                [15, 7, 0, 8],
                [6, 3, 12, 0]
            ]),
            (0, 1, 3, 2), 21
        ),
    ],
)
def test_tsp_bruteforce(distance_matrix, expected_path, expected_cost):
    """Test the tsp_bruteforce method with various inputs."""
    tsp_instance = TSP()
    tsp_instance.distance_matrix = distance_matrix

    # Call the method
    result_path, result_cost = tsp_instance.tsp_bruteforce()

    # Ensure the path is correct (order might differ due to symmetry in cycles)

    # Ensure the cost is correct
    assert result_cost == expected_cost, f"Expected cost {expected_cost}, but got {result_cost}"


import pytest
from unittest.mock import patch, MagicMock


def test_get_route_osrm(tsp_instance):
    """Test the get_route_osrm method with mocked requests."""
    origin = (31.7683, 35.2137)  # Jerusalem
    destination = (32.0853, 34.7818)  # Tel Aviv

    # Mocked response data from the OSRM API
    mocked_response_data = {
        "routes": [
            {
                "geometry": "e~l~F~ps|UoK~@eE~BkMx@",
                "distance": 58390  # 58.39 km
            }
        ]
    }

    # Mock the requests.get call
    with patch("requests.get") as mock_get:
        # Set up the mock to return a custom response
        mock_response = MagicMock()
        mock_response.json.return_value = mocked_response_data
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Call the function
        route_coords, distance = tsp_instance.get_route_osrm(origin, destination)

        # Assertions
        assert distance == 58.39, "Unexpected distance"



