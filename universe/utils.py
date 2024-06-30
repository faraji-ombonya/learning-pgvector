from pgvector.django import CosineDistance
from universe.models import File

def search_image_embedding(self, embedding):
    user_files = File.objects.filter()

    files_with_distance = user_files.annotate(
        distance = CosineDistance("embedding_clip_vit_l_14", embedding)
    ).order_by("distance")[:12]