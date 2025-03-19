from manim import *
import numpy as np

# Set a white background
config.background_color = WHITE


class CustomArrow3D(Line):
    """A custom 3D arrow implementation that works with Manim Community v0.19.0"""

    def __init__(self, start, end, color=BLACK, thickness=0.02, **kwargs):
        if 'opacity' in kwargs:
            kwargs.pop('opacity')

        super().__init__(start, end, color=color, **kwargs)
        # Use a simple cone as tip instead of the standard tip
        cone = Cone(direction=self.get_unit_vector(), base_radius=thickness * 3, height=thickness * 8)
        cone.set_color(color)
        cone.shift(end - cone.get_direction() * cone.height / 2)
        self.add(cone)


class TransformerEncodingAnimation(ThreeDScene):
    def construct(self):
        # Set up the scene
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.1)

        # Create 3D axes with dark colors for white background
        axes = ThreeDAxes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            z_range=[-5, 5, 1],
            x_length=10,
            y_length=10,
            z_length=10,
            color=DARK_GREY
        )

        # Add labels with dark colors - create Text objects instead of using color parameter
        x_label = Text("Dimension 1", color=BLACK, font_size=20)
        x_label.next_to(axes.x_axis.get_end(), RIGHT)

        y_label = Text("Dimension 2", color=BLACK, font_size=20)
        y_label.next_to(axes.y_axis.get_end(), UP)

        z_label = Text("Dimension 3", color=BLACK, font_size=20)
        z_label.next_to(axes.z_axis.get_end(), OUT)
        z_label.rotate(PI / 2, axis=RIGHT)

        # Title
        title = Text("Transformer Encoding in Vector Space", color=BLACK, font_size=24).to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)

        # Setup for transformer layers
        num_layers = 5
        layer_names = [f"Layer {i + 1}" for i in range(num_layers)]
        colors = [BLUE_E, BLUE_C, BLUE_B, BLUE_A, BLUE]

        # Initial zero vector and vector accumulation
        origin = axes.coords_to_point(0, 0, 0)
        start_point = Sphere(radius=0.1).set_color(BLACK).move_to(origin)  # Use Sphere instead of Dot3D

        # Legend for layers
        legend = VGroup()
        for i, (name, color) in enumerate(zip(layer_names, colors)):
            dot = Dot(color=color, radius=0.1)
            label = Text(name, color=color, font_size=24)
            item = VGroup(dot, label).arrange(RIGHT, buff=0.2)
            legend.add(item)

        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        legend.scale(0.6).to_corner(UR)
        self.add_fixed_in_frame_mobjects(legend)

        # Add everything to the scene
        self.play(
            Create(axes),
            Write(x_label),
            Write(y_label),
            Write(z_label),
            FadeIn(title),
            FadeIn(start_point),
        )
        self.wait()

        # Initial vector components for each layer (will be summed up)
        layer_components = [
            np.array([1.5, 0.5, 0.2]),  # Layer 1: basic syntactic information
            np.array([0.5, 1.2, 0.8]),  # Layer 2: word-level semantics
            np.array([0.2, 0.8, 1.5]),  # Layer 3: local context integration
            np.array([0.8, 0.3, 1.0]),  # Layer 4: higher-level patterns
            np.array([0.5, 0.7, 0.7]),  # Layer 5: refined representation
        ]

        # Accumulate vectors
        accumulated_vector = np.zeros(3)
        prev_vector_end = origin
        all_vectors = []
        all_labels = []

        for i, component in enumerate(layer_components):
            # Update the accumulated vector
            accumulated_vector += component

            # Convert to Manim coordinates
            vec_end = axes.coords_to_point(*accumulated_vector)

            # Create the vector arrow
            vector = CustomArrow3D(
                prev_vector_end,
                vec_end,
                color=colors[i],
                thickness=0.03
            )

            # Create a label for this contribution
            contribution_label = Text(f"{layer_names[i]} Contribution",
                                      color=colors[i],
                                      font_size=20)
            contribution_label.next_to(vector.get_center(), UP + RIGHT)
            self.add_fixed_orientation_mobjects(contribution_label)
            all_labels.append(contribution_label)

            # Animate the addition of this vector component
            self.play(
                Create(vector),  # Use Create instead of GrowArrow
                FadeIn(contribution_label),
                run_time=1.5
            )
            self.wait(0.5)

            # Save the vector for future reference
            all_vectors.append(vector)

            # Update the previous vector end for the next component
            prev_vector_end = vec_end

        # Final vector from origin to end point (representing the complete embedding)
        final_vector = CustomArrow3D(
            origin,
            vec_end,
            color=BLACK,
            thickness=0.05
        )

        final_label = Text("Final Encoded Vector", color=BLACK, font_size=24)
        final_label.next_to(final_vector.get_center(), RIGHT + UP)
        self.add_fixed_orientation_mobjects(final_label)

        # Highlight the final vector
        self.play(
            Create(final_vector),
            FadeIn(final_label),
            *[vector.animate.set_opacity(0.3) for vector in all_vectors],
            *[label.animate.set_opacity(0.3) for label in all_labels],
            run_time=2
        )

        # Add an attention visualization
        self.wait()
        self.play(*[FadeOut(label) for label in all_labels])

        # Show attention between token vectors
        token_positions = [
            np.array([2, 2, 2]),
            np.array([3, 1, 3]),
            np.array([1, 3, 2]),
            np.array([2, 1, 4]),
        ]

        token_dots = []
        token_labels = []

        for i, pos in enumerate(token_positions):
            dot = Sphere(radius=0.1).set_color(GOLD).move_to(axes.coords_to_point(*pos))
            label = Text(f"Token {i + 1}", color=DARK_BROWN, font_size=20)
            label.next_to(dot, UP)
            self.add_fixed_orientation_mobjects(label)
            token_dots.append(dot)
            token_labels.append(label)

        self.play(
            *[FadeIn(dot) for dot in token_dots],
            *[FadeIn(label) for label in token_labels],
            final_vector.animate.set_opacity(0.4),
            run_time=1.5
        )

        # Create attention lines between tokens - Fixed for opacity issue
        attention_lines = []
        for i in range(len(token_positions)):
            for j in range(len(token_positions)):
                if i != j:
                    start = axes.coords_to_point(*token_positions[i])
                    end = axes.coords_to_point(*token_positions[j])
                    # Create a regular line and set its stroke opacity instead of using opacity parameter
                    line = DashedLine(start, end, color=RED)
                    line.set_stroke(opacity=0.6)
                    attention_lines.append(line)

        self.play(
            *[Create(line) for line in attention_lines],
            run_time=2
        )

        # Show context integration
        context_sphere = Sphere(radius=0.5)
        context_sphere.set_color(PURPLE)
        context_sphere.set_fill(opacity=0.3)  # Set fill opacity instead of overall opacity
        context_sphere.move_to(axes.coords_to_point(*accumulated_vector))

        context_label = Text("Contextual Representation", color=PURPLE, font_size=20)
        context_label.next_to(context_sphere, RIGHT)
        self.add_fixed_orientation_mobjects(context_label)

        self.play(
            FadeIn(context_sphere),
            FadeIn(context_label),
            run_time=1.5
        )

        # Rotate the camera to show the 3D nature
        self.wait(5)

        # Fade everything out
        self.play(
            *[FadeOut(obj) for obj in [*all_vectors, final_vector, *token_dots, *token_labels,
                                       *attention_lines, context_sphere, context_label, final_label]],
            run_time=2
        )
        self.wait()


class SelfAttentionMechanism(ThreeDScene):
    def construct(self):
        # Set up camera
        self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)

        # Create 3D axes with dark colors for white background
        axes = ThreeDAxes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            z_range=[-5, 5, 1],
            x_length=10,
            y_length=10,
            z_length=10,
            color=DARK_GREY
        )

        # Title
        title = Text("Self-Attention Mechanism", color=BLACK, font_size=24).to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)

        # Add the axes to the scene
        self.play(Create(axes), Write(title))

        # Create token embeddings
        num_tokens = 4
        token_colors = [BLUE, GREEN, GOLD, RED]
        token_positions = [
            np.array([2, 1, 1]),
            np.array([1, 3, 0]),
            np.array([0, 1, 3]),
            np.array([3, 2, 2]),
        ]

        token_dots = []
        token_labels = []

        for i, (pos, color) in enumerate(zip(token_positions, token_colors)):
            # Convert to Manim coordinates
            position = axes.coords_to_point(*pos)

            # Create a dot for the token
            dot = Sphere(radius=0.1).set_color(color).move_to(position)
            token_dots.append(dot)

            # Create a label for the token
            label = Text(f"Token {i + 1}", color=color, font_size=20)
            label.next_to(dot, UP)
            self.add_fixed_orientation_mobjects(label)
            token_labels.append(label)

        # Add token dots and labels to the scene
        self.play(
            *[FadeIn(dot) for dot in token_dots],
            *[FadeIn(label) for label in token_labels],
            run_time=1.5
        )

        # Create query, key, and value vectors for each token
        q_arrows = []
        k_arrows = []
        v_arrows = []

        q_label = Text("Query Vectors (Q)", color=PURPLE, font_size=20).to_corner(UR)
        k_label = Text("Key Vectors (K)", color=ORANGE, font_size=20).next_to(q_label, DOWN)
        v_label = Text("Value Vectors (V)", color=TEAL, font_size=20).next_to(k_label, DOWN)

        self.add_fixed_in_frame_mobjects(q_label, k_label, v_label)
        self.play(Write(q_label), Write(k_label), Write(v_label))

        # Define vector directions for Q, K, V for each token
        q_directions = [
            np.array([0.5, 0.3, 0.7]),
            np.array([0.2, 0.8, 0.3]),
            np.array([0.7, 0.2, 0.4]),
            np.array([0.4, 0.6, 0.5]),
        ]

        k_directions = [
            np.array([0.6, 0.4, 0.3]),
            np.array([0.3, 0.7, 0.5]),
            np.array([0.5, 0.3, 0.7]),
            np.array([0.7, 0.5, 0.2]),
        ]

        v_directions = [
            np.array([0.3, 0.5, 0.6]),
            np.array([0.7, 0.2, 0.5]),
            np.array([0.4, 0.7, 0.3]),
            np.array([0.5, 0.4, 0.6]),
        ]

        # Scaling factor for the vectors
        scale = 1.5

        # Create and animate the query vectors
        for i, (dot, direction) in enumerate(zip(token_dots, q_directions)):
            start = dot.get_center()
            end = start + scale * direction

            arrow = CustomArrow3D(start, end, color=PURPLE)
            q_arrows.append(arrow)

        self.play(*[Create(arrow) for arrow in q_arrows], run_time=1.5)
        self.wait(0.5)

        # Create and animate the key vectors
        for i, (dot, direction) in enumerate(zip(token_dots, k_directions)):
            start = dot.get_center()
            end = start + scale * direction

            arrow = CustomArrow3D(start, end, color=ORANGE)
            k_arrows.append(arrow)

        self.play(*[Create(arrow) for arrow in k_arrows], run_time=1.5)
        self.wait(0.5)

        # Create and animate the value vectors
        for i, (dot, direction) in enumerate(zip(token_dots, v_directions)):
            start = dot.get_center()
            end = start + scale * direction

            arrow = CustomArrow3D(start, end, color=TEAL)
            v_arrows.append(arrow)

        self.play(*[Create(arrow) for arrow in v_arrows], run_time=1.5)
        self.wait()

        # Explain attention scores calculation
        attention_title = Text("Computing Attention Scores: Q·K^T", color=BLACK, font_size=24).to_corner(UL)
        self.add_fixed_in_frame_mobjects(attention_title)
        self.play(Transform(title, attention_title))

        # Show attention between tokens (token 1 attends to all others)
        attention_lines = []
        attention_scores = []

        focus_token = 0  # We'll focus on Token 1's attention to others

        for i in range(num_tokens):
            if i != focus_token:
                # Get dot product of query and key (simulate with random values)
                # In reality, this would be Q·K^T
                score_value = np.random.uniform(0.3, 0.9)

                # Create a line connecting token 1's query to token i's key
                q_end = token_dots[focus_token].get_center() + scale * q_directions[focus_token]
                k_end = token_dots[i].get_center() + scale * k_directions[i]

                # Fixed: Use stroke opacity instead of overall opacity
                line = DashedLine(q_end, k_end, color=DARK_GREY)
                line.set_stroke(opacity=score_value)
                attention_lines.append(line)

                # Create a label for the attention score
                score_label = Text(f"{score_value:.2f}", color=BLACK, font_size=18)
                score_label.move_to((q_end + k_end) / 2 + 0.3 * UP)
                self.add_fixed_orientation_mobjects(score_label)
                attention_scores.append(score_label)

        self.play(
            *[Create(line) for line in attention_lines],
            *[Write(score) for score in attention_scores],
            run_time=2
        )
        self.wait()

        # Show weighted value vectors
        weighted_v_title = Text("Weighted Value Vectors", color=BLACK, font_size=24).to_corner(UL)
        self.add_fixed_in_frame_mobjects(weighted_v_title)
        self.play(Transform(title, weighted_v_title))

        # Create weighted value vectors
        weighted_v_arrows = []

        for i in range(num_tokens):
            if i != focus_token:
                # Get the score value (parsed from the text)
                idx = i if i < focus_token else i - 1
                score_text = attention_scores[idx].text
                score_value = float(score_text)

                # Get the start and end points of the value vector
                start = token_dots[i].get_center()

                # Create a new scaled vector based on attention score
                weighted_end = start + scale * score_value * v_directions[i]

                arrow = CustomArrow3D(start, weighted_end, color=GREEN)
                weighted_v_arrows.append(arrow)

        self.play(*[Create(arrow) for arrow in weighted_v_arrows], run_time=1.5)
        self.wait()

        # Show the final output vector (sum of weighted values)
        output_title = Text("Final Output Vector (Token 1)", color=BLACK, font_size=24).to_corner(UL)
        self.add_fixed_in_frame_mobjects(output_title)
        self.play(Transform(title, output_title))

        # Calculate the sum of weighted value vectors (simplified)
        output_vector = np.zeros(3)
        for i in range(num_tokens):
            if i != focus_token:
                idx = i if i < focus_token else i - 1
                score_text = attention_scores[idx].text
                score_value = float(score_text)
                output_vector += score_value * v_directions[i]

        # Normalize and scale
        norm = np.linalg.norm(output_vector)
        if norm > 0:  # Avoid division by zero
            output_vector = 2 * output_vector / norm

        # Create the output vector
        start = token_dots[focus_token].get_center()
        end = start + scale * output_vector

        output_arrow = CustomArrow3D(start, end, color=RED_E, thickness=0.05)

        output_label = Text("Output", color=RED_E, font_size=24)
        output_label.next_to(output_arrow, RIGHT)
        self.add_fixed_orientation_mobjects(output_label)

        # Fixed opacity issue for animate method
        for arrow in q_arrows + k_arrows + v_arrows + weighted_v_arrows:
            arrow.set_opacity(0.3)

        for line in attention_lines:
            line.set_stroke(opacity=0.3)

        self.play(
            Create(output_arrow),
            Write(output_label),
            run_time=2
        )

        # Final view
        self.wait(5)

        # Fade out
        self.play(
            *[FadeOut(obj) for obj in [*token_dots, *token_labels, *q_arrows, *k_arrows, *v_arrows,
                                       *weighted_v_arrows, *attention_lines, *attention_scores,
                                       output_arrow, output_label]],
            run_time=2
        )
        self.wait()
