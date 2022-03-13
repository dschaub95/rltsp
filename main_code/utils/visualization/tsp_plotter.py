import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class TSP_plotter:
    def __init__(self) -> None:
        pass

    def plot_nx_graph(
        self,
        graph,
        draw_edges=True,
        opt_solution=None,
        opt_len=None,
        pred_solution=None,
        pred_len=None,
        partial_solution=[],
        title="",
        edge_probs=None,
        save_path=None,
        node_values=None,
        only_draw_relevant_edges=False,
        dpi=200,
        show=True,
    ):
        plt.style.use("seaborn-paper")
        cmap = plt.cm.plasma.reversed()
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), sharex=True, sharey=True, dpi=dpi)
        if node_values is not None:
            nodes_sizes = (
                np.exp(node_values) / np.sum(np.exp(node_values)) + 0.2
            ) * 500
        else:
            nodes_sizes = 200
        edge_list = [edge for edge in graph.edges]
        if opt_solution is not None:
            opt_tour_edges = list(zip(opt_solution, opt_solution[1:]))
            opt_tour_edges.append((opt_solution[-1], opt_solution[0]))
        if pred_solution is not None:
            pred_tour_edges = list(zip(pred_solution, pred_solution[1:]))
            pred_tour_edges.append((pred_solution[-1], pred_solution[0]))
        if len(partial_solution) > 1:
            partial_tour_edges = list(zip(partial_solution, partial_solution[1:]))

        labels = {i: i for i in graph.nodes}
        pos = {i: graph.nodes[i]["coord"] for i in graph.nodes}
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color="y", node_size=nodes_sizes)
        if len(partial_solution) > 0:
            # mark last selected node
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=[partial_solution[-1]],
                ax=ax,
                node_color="r",
                node_size=200,
                label="Last selected node",
            )
        if only_draw_relevant_edges and len(partial_solution) > 1:
            # keep edges of partial tour
            remaining_edges_partial = [
                edge
                for edge in edge_list
                if edge in partial_tour_edges or edge[::-1] in partial_tour_edges
            ]
            # keep edges of optimal solution
            remaining_edges_opt = [
                edge
                for edge in edge_list
                if edge in opt_tour_edges or edge[::-1] in opt_tour_edges
            ]
            # delete all edges that are not relevant during the next selection step
            last_node = partial_solution[-1]
            remaining_edges_select = [
                edge
                for edge in edge_list
                if (edge[0] == last_node and edge[1] not in partial_solution)
                or (edge[1] == last_node and edge[0] not in partial_solution)
            ]
            edge_list = (
                remaining_edges_partial + remaining_edges_opt + remaining_edges_select
            )
            if edge_probs is not None:

                probabilties = [edge_probs[edge[0], edge[1]] for edge in edge_list]
                edge_colors = (np.array(probabilties) + 0.05) * 50
                edge_alphas = [0.1 if prob == 0.0 else 0.5 for prob in probabilties]
                edges = nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=edge_list,
                    ax=ax,
                    edge_color=edge_colors,
                    width=1,
                    alpha=1.0,
                    edge_cmap=cmap,
                )
            else:
                nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=edge_list,
                    ax=ax,
                    edge_color="y",
                    width=1,
                    alpha=1.0,
                )
        else:
            if edge_probs is not None:
                # edge_list = np.transpose(np.array(np.where(np.triu(edge_probs) > 0)))
                # edge_list = np.transpose(np.asarray(np.triu(edge_probs) > 0).nonzero())
                probabilties = [edge_probs[edge[0], edge[1]] for edge in edge_list]
                edge_colors = (np.array(probabilties) + 0.05) * 50
                edge_alphas = [0.1 if prob == 0.0 else 0.5 for prob in probabilties]
                edges = nx.draw_networkx_edges(
                    graph,
                    pos,
                    edgelist=edge_list,
                    ax=ax,
                    edge_color=edge_colors,
                    width=1,
                    alpha=0.2,
                    edge_cmap=cmap,
                )
            else:
                nx.draw_networkx_edges(
                    graph, pos, ax=ax, edge_color="y", width=1, alpha=0.1
                )
        if opt_solution is not None:
            if opt_len is not None:
                label = f"Optimal solution, dist: {opt_len:.5}"
            else:
                label = "Optimal solution"
            nx.draw_networkx_edges(
                graph,
                pos,
                ax=ax,
                edgelist=opt_tour_edges,
                edge_color="r",
                label=label,
                width=1,
                # style=(0, (5, 10)),
                style=":",
            )
        if pred_solution is not None:
            if pred_len is not None:
                label = f"Predicted solution, dist: {pred_len:.5}"
            else:
                label = "Predicted solution"
            nx.draw_networkx_edges(
                graph,
                pos,
                ax=ax,
                edgelist=pred_tour_edges,
                edge_color="g",
                label=label,
                width=1,
                # style=(0, (5, 10)),
                style=":",
            )
        if len(partial_solution) > 1:
            nx.draw_networkx_edges(
                graph,
                pos,
                ax=ax,
                edgelist=partial_tour_edges,
                edge_color="black",
                label="Current solution",
                width=1,
                style=(0, (1, 10)),
                # style=":",
            )
        # Draw labels
        nx.draw_networkx_labels(graph, pos, ax=ax, labels=labels, font_size=9)
        # ax.set(xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))

        # set alpha value for each edge
        # for i in range(len(edge_list)):
        #     edges[i].set_alpha(edge_alphas[i])

        # pc = mpl.collections.PatchCollection(edges, cmap=cmap)
        # pc.set_array(edge_colors)
        # plt.colorbar(pc)

        ax.legend()
        ax.set_xlabel("x-coordinate")
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        ax.set_ylabel("y-coordinate")
        ax.set_title(title)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, format="raw", dpi=300)
        if show:
            plt.show()
        return fig
