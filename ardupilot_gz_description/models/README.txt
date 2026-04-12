Aerial texture for Gazebo

The aerial orthophoto is stored compressed so the repo stays under GitHub size limits.

After clone or pull, from model texture directory run:

  gunzip -k edinburgh-19_aerial.png.gz

That writes edinburgh-19_aerial.png next to the .gz file (GNU gunzip -k keeps the archive).
model.sdf expects that PNG path.

Note: the archived PNG is half linear resolution (6272 x 6272) versus a typical full export,
to keep the compressed artifact under ~100 MB. For simulation it is usually enough for the
heightmap diffuse overlay.
