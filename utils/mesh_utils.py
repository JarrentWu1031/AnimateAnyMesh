import torch
import numpy as np
import time

def get_adjacency_matrix(vertices, faces, valid_len):
    """
    Args:
        vertices: [B, N, 3] tensor
        faces: [B, M, 3] tensor, padded with -1
        valid_len: [B] tensor
    Returns:
        adj_matrix: [B, N, N] tensor
    """
    B, N, _ = vertices.shape
    device = vertices.device
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, faces.size(1))  # [B, M]
    valid_faces_mask = (faces >= 0).all(dim=-1)  # [B, M]
    edges_0 = torch.stack([faces[..., 0], faces[..., 1]], dim=-1)  # [B, M, 2]
    edges_1 = torch.stack([faces[..., 1], faces[..., 2]], dim=-1)  # [B, M, 2]
    edges_2 = torch.stack([faces[..., 2], faces[..., 0]], dim=-1)  # [B, M, 2]
    edges = torch.cat([edges_0, edges_1, edges_2], dim=1)  # [B, 3M, 2]
    batch_idx = batch_idx.unsqueeze(-1).expand(-1, -1, 2)  # [B, M, 2]
    batch_idx = torch.cat([batch_idx, batch_idx, batch_idx], dim=1)  # [B, 3M, 2]
    valid_mask = valid_faces_mask.unsqueeze(-1).expand(-1, -1, 2)  # [B, M, 2]
    valid_mask = torch.cat([valid_mask, valid_mask, valid_mask], dim=1)  # [B, 3M, 2]
    flat_edges = edges[valid_mask].view(-1, 2)  # [Valid_edges, 2]
    flat_batch = batch_idx[valid_mask].view(-1, 2)[:, 0]  # [Valid_edges]
    indices = torch.stack([
        flat_batch,
        flat_edges[:, 0],
        flat_edges[:, 1]
    ], dim=0)  # [3, Valid_edges]
    values = torch.ones(indices.size(1), device=device)
    adj = torch.sparse_coo_tensor(
        indices, 
        values, 
        (B, N, N),
        device=device
    ).to_dense()
    adj = (adj + adj.transpose(1, 2)) > 0
    mask = torch.arange(N, device=device)[None, :] < valid_len[:, None]  # [B, N]
    mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B, N, N]
    adj = adj & mask
    return adj

def mesh_preprocess(vertices, faces, max_length=4096):
    if vertices.ndim == 4:
        vertices, faces = vertices[0], faces[0]
    center = (vertices[0].max(dim=0)[0] + vertices[0].min(dim=0)[0]) / 2
    vertices = vertices - center
    v_max = vertices[0].abs().max()
    vertices = vertices / v_max
    valid_mask = torch.ones(vertices.shape[1], dtype=torch.bool)
    faces = torch.tensor(faces, dtype=torch.int64)
    faces_max_length = int(max_length * 2.5)
    vertices = torch.cat([vertices, torch.zeros(vertices.shape[0], max_length-vertices.shape[1], 3)], dim=1)
    vertices_color = torch.cat([vertices_color, -1.0 * torch.ones(max_length-vertices_color.shape[0], 3)], dim=0)
    faces = torch.cat([faces, -1 * torch.ones(faces_max_length-faces.shape[0], 3).to(torch.int64)], dim=0)
    valid_mask = torch.cat([valid_mask, torch.zeros(max_length-valid_mask.shape[0], dtype=torch.bool)])[None]
    valid_length = valid_mask.sum(dim=-1)
    adj_matrix = get_adjacency_matrix(vertices[0][None], faces[None], valid_length)
    return vertices, vertices_color, faces, valid_mask, valid_length, adj_matrix

def merge_identical_vertices(vertices, faces):
    """
    Merge identical vertices
    """
    rounded_vertices = vertices
    _, unique_indices, inverse_indices = np.unique(
        rounded_vertices.view([('', rounded_vertices.dtype)]*3),
        return_index=True,
        return_inverse=True
    )
    merged_vertices = vertices[unique_indices]
    merged_faces = inverse_indices[faces]
    max_valid_index = len(unique_indices) - 1
    valid_faces_mask = (merged_faces <= max_valid_index).all(axis=1)
    merged_faces = merged_faces[valid_faces_mask]
    valid_faces_mask = np.apply_along_axis(lambda x: len(np.unique(x)), 1, merged_faces) == 3
    merged_faces = merged_faces[valid_faces_mask]
    sorted_faces = np.sort(merged_faces, axis=1)
    _, unique_face_idx = np.unique(sorted_faces.view([('', sorted_faces.dtype)]*3),
                                 return_index=True)
    merged_faces = merged_faces[unique_face_idx]
    assert merged_faces.max() < len(unique_indices), "Face indices out of bounds"
    assert merged_faces.min() >= 0, "Negative face indices found"
    return merged_vertices, merged_faces

def find_indices_in_merged(vertices_list, merged_vertices):
    indices_list = []
    for vertices in vertices_list:
        matches = (merged_vertices.unsqueeze(1) == vertices.unsqueeze(0))
        matches = matches.all(dim=2)  
        indices = matches.nonzero()[:, 0]
        indices = indices.reshape(vertices.shape[0])
        indices_list.append(indices)
    return indices_list

def merge_identical_vertices_with_indices(vertices_list, faces_list):
    """
    Args:
        vertices_list: list of torch.Tensor[(Ni,3)]
        faces_list: list of torch.Tensor[(Fi,3)]
    Returns:
        merged_vertices: torch.Tensor 
        merged_faces: torch.Tensor 
        indices_list: list of torch.Tensor 
    """
    all_vertices = torch.cat(vertices_list, dim=0)
    all_faces = torch.cat(faces_list, dim=0)
    vertices_tuple = torch.stack(all_vertices.unbind(dim=-1), dim=-1)
    unique_vertices, inverse_indices = torch.unique(
        vertices_tuple,
        dim=0,
        return_inverse=True,
        sorted=True
    )
    merged_faces = inverse_indices[all_faces]
    face_vertices_equal = (merged_faces[:,[0,0,1]] == merged_faces[:,[1,2,2]]).any(dim=1)
    valid_faces = ~face_vertices_equal
    merged_faces = merged_faces[valid_faces]
    sorted_faces, _ = torch.sort(merged_faces, dim=1)
    faces_tuple = torch.stack(sorted_faces.unbind(dim=-1), dim=-1)
    merged_faces = torch.unique(faces_tuple, dim=0, sorted=True)
    start_idx = 0
    indices_list = []
    for vertices in vertices_list:
        indices_list.append(inverse_indices[start_idx:start_idx + len(vertices)])
        start_idx += len(vertices)
    assert merged_faces.max() < len(unique_vertices), "Face indices out of bounds"
    assert merged_faces.min() >= 0, "Negative face indices found"
    return unique_vertices, merged_faces, indices_list
