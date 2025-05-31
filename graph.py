"""
Graf veri yapısı - A* algoritması için
Hazır fonksiyon kullanmadan yazılmıştır
"""

class Node:
    """Graf düğümü sınıfı"""
    
    def __init__(self, x, y, node_type='empty'):
        self.x = x
        self.y = y
        self.node_type = node_type  # 'empty', 'wall', 'start', 'goal'
        self.g_cost = float('inf')  # Başlangıçtan bu düğüme maliyet
        self.h_cost = 0  # Heuristic maliyet (hedefe tahmini)
        self.f_cost = float('inf')  # g + h
        self.parent = None  # Yol takibi için
        self.visited = False
        self.in_open_set = False
    
    def __lt__(self, other):
        """Priority queue için karşılaştırma"""
        if self.f_cost == other.f_cost:
            return self.h_cost < other.h_cost
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def reset(self):
        """Düğümü algoritma için sıfırla"""
        self.g_cost = float('inf')
        self.h_cost = 0
        self.f_cost = float('inf')
        self.parent = None
        self.visited = False
        self.in_open_set = False


class Graph:
    """2D Grid tabanlı graf sınıfı"""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.nodes = {}
        self.start_node = None
        self.goal_node = None
        self.create_grid()
    
    def create_grid(self):
        """Grid oluştur"""
        for x in range(self.width):
            for y in range(self.height):
                self.nodes[(x, y)] = Node(x, y)
    
    def get_node(self, x, y):
        """Koordinatlara göre düğüm getir"""
        if (x, y) in self.nodes:
            return self.nodes[(x, y)]
        return None
    
    def set_start(self, x, y):
        """Başlangıç düğümünü ayarla"""
        if self.start_node:
            self.start_node.node_type = 'empty'
        
        node = self.get_node(x, y)
        if node:
            node.node_type = 'start'
            self.start_node = node
    
    def set_goal(self, x, y):
        """Hedef düğümünü ayarla"""
        if self.goal_node:
            self.goal_node.node_type = 'empty'
        
        node = self.get_node(x, y)
        if node:
            node.node_type = 'goal'
            self.goal_node = node
    
    def set_wall(self, x, y):
        """Duvar ekle"""
        node = self.get_node(x, y)
        if node and node.node_type == 'empty':
            node.node_type = 'wall'
    
    def remove_wall(self, x, y):
        """Duvarı kaldır"""
        node = self.get_node(x, y)
        if node and node.node_type == 'wall':
            node.node_type = 'empty'
    
    def get_neighbors(self, node):
        """Düğümün komşularını getir (8 yön)"""
        neighbors = []
        directions = [
            (-1, -1), (-1, 0), (-1, 1),  # Üst
            (0, -1),           (0, 1),   # Yan
            (1, -1),  (1, 0),  (1, 1)   # Alt
        ]
        
        for dx, dy in directions:
            new_x, new_y = node.x + dx, node.y + dy
            neighbor = self.get_node(new_x, new_y)
            
            if neighbor and neighbor.node_type != 'wall':
                neighbors.append(neighbor)
        
        return neighbors
    
    def get_distance(self, node1, node2):
        """İki düğüm arası gerçek mesafe"""
        dx = abs(node1.x - node2.x)
        dy = abs(node1.y - node2.y)
        
        # Çapraz hareket için Euclidean benzeri
        if dx > 0 and dy > 0:
            return 14  # Yaklaşık sqrt(2) * 10
        else:
            return 10  # Düz hareket
    
    def reset_all_nodes(self):
        """Tüm düğümleri algoritma için sıfırla"""
        for node in self.nodes.values():
            node.reset()
    
    def create_random_walls(self, wall_percentage=0.3):
        """Rastgele duvarlar oluştur"""
        import random
        
        total_nodes = self.width * self.height
        wall_count = int(total_nodes * wall_percentage)
        
        added_walls = 0
        while added_walls < wall_count:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            
            node = self.get_node(x, y)
            if node and node.node_type == 'empty':
                node.node_type = 'wall'
                added_walls += 1
    
    def create_maze_pattern(self):
        """Labirent benzeri desen oluştur"""
        # Basit labirent pattern
        for x in range(0, self.width, 4):
            for y in range(self.height):
                if y % 4 != 0:
                    self.set_wall(x, y)
        
        for y in range(0, self.height, 4):
            for x in range(self.width):
                if x % 4 != 0:
                    self.set_wall(x, y)
    
    def get_total_nodes(self):
        """Toplam düğüm sayısını döndür"""
        return len(self.nodes)
    
    def get_empty_nodes_count(self):
        """Boş düğüm sayısını döndür"""
        count = 0
        for node in self.nodes.values():
            if node.node_type == 'empty':
                count += 1
        return count