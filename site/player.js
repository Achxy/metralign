const frame = document.querySelector(".video-frame");
const canEmbed = window.location.protocol === "http:" || window.location.protocol === "https:";

if (frame && canEmbed) {
  const player = document.createElement("iframe");
  const videoId = frame.dataset.videoId;

  player.src = `https://www.youtube-nocookie.com/embed/${videoId}?rel=0`;
  player.title = "Metralign project explanation";
  player.allow =
    "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share";
  player.referrerPolicy = "strict-origin-when-cross-origin";
  player.allowFullscreen = true;

  frame.replaceChildren(player);
}
