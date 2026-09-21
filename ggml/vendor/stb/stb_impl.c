/* stb_image, compiled once. The file reader is ours (STBI_NO_STDIO), so stb only
   ever sees memory and never touches stdio. PNG is all this project needs. */
#define STBI_NO_STDIO
#define STBI_ONLY_PNG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
