#include <SFML/Graphics.hpp>

void offScreenDrawing() {
  sf::RenderTexture renderTexture;
  if (!renderTexture.create(500, 500)) {
    // error...
  }

  renderTexture.clear();
  // renderTexture.draw(sprite);
  renderTexture.display();

  const sf::Texture& texture = renderTexture.getTexture();

  sf::Sprite sprite(texture);
  //   window.draw(sprite);
}

// int main() {
//   sf::RenderWindow window(sf::VideoMode(800, 600), "My window");
//   sf::RenderTexture renderTexture;

//   while (window.isOpen()) {
//     sf::Event event;
//     while (window.pollEvent(event)) {
//       if (event.type == sf::Event::Closed)
//         window.close();
//     }

//     window.clear(sf::Color::Black);

//     const sf::Texture& texture = renderTexture.getTexture();
//     sf::Sprite sprite(texture);
//     window.draw(sprite);

//     window.display();
//   }

//   return 0;
// }

// void renderingThread(sf::RenderWindow* window) {
//   window->setActive(true);

//   while (window->isOpen()) {
//     window->display();
//   }
// }

// int main() {
//   sf::RenderWindow window(sf::VideoMode(800, 600), "OpenGL");

//   window.setActive(false);
//   sf::Thread thread(&renderingThread, &window);
//   thread.launch();

//   while (window.isOpen()) {
//     //
//   }

//   return 0;
// }
#include <iostream>

int main() {
  sf::Texture texture;
  std::string filename = "test.jpg";
  if (!texture.loadFromFile(filename)) {
    std::cerr << "Cannot load file: " << filename << '\n';
  }

  sf::RenderWindow window(sf::VideoMode(800, 600), "My window");

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed)
        window.close();

      window.clear(sf::Color::Black);
      sf::Sprite sprite(texture);
      // sprite.setTextureRect(sf::IntRect(10, 10, 32, 32));
      // sprite.setColor(sf::Color(0, 255, 0));

      // absolution position
      // sprite.setPosition(sf::Vector2f(10.f, 50.f));
      // sprite.move(sf::Vector2f(5.f, 10.f));
      // sprite.setRotation(90.0f);
      sprite.rotate(15.f);

      // sprite.setScale(sf::Vector2f(0.5f, 2.f));
      sprite.scale(sf::Vector2f(0.25f, 0.25f));

      window.draw(sprite);
      window.display();
    }
  }
}
