from django.db import models
from pgvector.django import VectorField, HnswIndex

class File(models.Model):
    name = models.CharField(null=True, blank=True)
    embedding_clip_vit_l_14 = VectorField(
        dimensions=768, 
        help_text="Vector embeddings (clip-vit-large-patch14) of the file content",
        null=True,
        blank=True,
    )


    class Meta:
        indexes = [
            HnswIndex(
                name="clip_l14_vectors_index",
                fields=["embedding_clip_vit_l_14"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"]
            )
        ]
