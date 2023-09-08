import tensorflow as tf
import arcade
from apple import Apple
from snake import Snake

game_width=800
game_height=560

class Game(arcade.Window):
    def __init__(self):
        super().__init__(width=game_width, height=game_height, title="Super Snake V1")
        arcade.set_background_color(arcade.color.KHAKI)

        self.snake=Snake(game_width,game_height)
        self.food=Apple(game_width,game_height)
        # self.model=tf.keras.models.load_model('my_snake_model.h5')

    def on_draw(self):
        arcade.start_render()
        self.snake.draw()
        self.food.draw()
        arcade.draw_text(f"score: {self.snake.score}", 10, 10, arcade.color.BLACK)
        arcade.finish_render()

    def on_update(self, delta_time: float):
        self.snake.move()

        data = {}
        output = self.model.predict(data)
        direction= output.argmax()
        if direction==0:
            self.snake.change_x = 0
            self.snake.change_y = 1
        elif direction==1:
            self.snake.change_x = 1
            self.snake.change_y = 0
        elif direction==2:
            self.snake.change_x = 0
            self.snake.change_y = -1
        elif direction==3:
            self.snake.change_x = -1
            self.snake.change_y = 0

        if arcade.check_for_collision(self.snake, self.food):
            self.snake.eat(self.food)
            self.food=Apple(game_width,game_height)

    def on_key_release(self, symbol: int, modifiers: int):
        pass

        
if __name__=="__main__":
    game=Game()
    arcade.run()