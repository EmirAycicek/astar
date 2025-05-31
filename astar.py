"""
A* Pathfinding Algoritması
Hazır fonksiyon kullanmadan yazılmıştır
"""

import heapq
from heuristics import HeuristicSelector


class AStar:
    """A* algoritması sınıfı"""
    
    def __init__(self, graph, heuristic_name='euclidean'):
        self.graph = graph
        self.heuristic_selector = HeuristicSelector()
        self.heuristic_func = self.heuristic_selector.get_heuristic(heuristic_name)
        self.heuristic_name = heuristic_name
        
        # Algoritma istatistikleri
        self.nodes_explored = 0
        self.nodes_in_open = 0
        self.path_length = 0
        self.algorithm_steps = []
        self.is_path_found = False
        
        # Animasyon için
        self.step_by_step = False
        self.current_step = 0
    
    def reset_stats(self):
        """İstatistikleri sıfırla"""
        self.nodes_explored = 0
        self.nodes_in_open = 0
        self.path_length = 0
        self.algorithm_steps = []
        self.is_path_found = False
        self.current_step = 0
    
    def calculate_heuristic(self, node, goal):
        """Heuristic değeri hesapla"""
        return self.heuristic_func(node, goal)
    
    def reconstruct_path(self, goal_node):
        """Yolu yeniden oluştur"""
        path = []
        current = goal_node
        
        while current is not None:
            path.append(current)
            current = current.parent
        
        path.reverse()
        return path
    
    def find_path(self, step_by_step=False):
        """
        A* algoritması ile yol bul
        
        Args:
            step_by_step: Adım adım çalışma modu
        
        Returns:
            tuple: (path, success, stats)
        """
        if not self.graph.start_node or not self.graph.goal_node:
            return [], False, "Başlangıç veya hedef düğüm belirlenmemiş"
        
        self.step_by_step = step_by_step
        self.reset_stats()
        self.graph.reset_all_nodes()
        
        start = self.graph.start_node
        goal = self.graph.goal_node
        
        # Open set (keşfedilecek düğümler) - Priority Queue
        open_set = []
        heapq.heappush(open_set, start)
        
        # Closed set (keşfedilmiş düğümler) - Set olarak
        closed_set = set()
        
        # Başlangıç düğümünü ayarla
        start.g_cost = 0
        start.h_cost = self.calculate_heuristic(start, goal)
        start.f_cost = start.g_cost + start.h_cost
        start.in_open_set = True
        
        step_count = 0
        
        while open_set:
            step_count += 1
            
            # En düşük f_cost'lu düğümü al
            current = heapq.heappop(open_set)
            current.in_open_set = False
            
            # Hedefe ulaştık mı?
            if current == goal:
                self.is_path_found = True
                path = self.reconstruct_path(current)
                self.path_length = len(path)
                
                # Son adımı kaydet
                if self.step_by_step:
                    self.algorithm_steps.append({
                        'step': step_count,
                        'current': current,
                        'action': 'goal_reached',
                        'path': path,
                        'open_set': list(open_set),
                        'closed_set': closed_set.copy()
                    })
                
                return path, True, self.get_stats()
            
            # Current'ı closed set'e ekle
            closed_set.add(current)
            current.visited = True
            self.nodes_explored += 1
            
            # Komşuları kontrol et
            neighbors = self.graph.get_neighbors(current)
            
            for neighbor in neighbors:
                # Zaten keşfedilmiş mi?
                if neighbor in closed_set:
                    continue
                
                # Yeni g_cost hesapla
                tentative_g_cost = current.g_cost + self.graph.get_distance(current, neighbor)
                
                # Bu yol daha iyi mi?
                if tentative_g_cost < neighbor.g_cost:
                    # Yolu güncelle
                    neighbor.parent = current
                    neighbor.g_cost = tentative_g_cost
                    neighbor.h_cost = self.calculate_heuristic(neighbor, goal)
                    neighbor.f_cost = neighbor.g_cost + neighbor.h_cost
                    
                    # Open set'te değilse ekle
                    if not neighbor.in_open_set:
                        heapq.heappush(open_set, neighbor)
                        neighbor.in_open_set = True
                        self.nodes_in_open += 1
            
            # Adım adım modda step bilgisi kaydet
            if self.step_by_step:
                self.algorithm_steps.append({
                    'step': step_count,
                    'current': current,
                    'action': 'exploring',
                    'neighbors': neighbors,
                    'open_set': list(open_set),
                    'closed_set': closed_set.copy(),
                    'open_count': len(open_set),
                    'closed_count': len(closed_set)
                })
        
        # Yol bulunamadı
        return [], False, self.get_stats()
    
    def get_stats(self):
        """Algoritma istatistiklerini döndür"""
        return {
            'nodes_explored': self.nodes_explored,
            'nodes_in_open': self.nodes_in_open,
            'path_length': self.path_length,
            'heuristic_used': self.heuristic_name,
            'path_found': self.is_path_found,
            'total_steps': len(self.algorithm_steps)
        }
    
    def get_step_info(self, step_index):
        """Belirli bir adımın bilgisini döndür"""
        if 0 <= step_index < len(self.algorithm_steps):
            return self.algorithm_steps[step_index]
        return None
    
    def get_total_steps(self):
        """Toplam adım sayısını döndür"""
        return len(self.algorithm_steps)
    
    def change_heuristic(self, heuristic_name):
        """Heuristic fonksiyonu değiştir"""
        self.heuristic_func = self.heuristic_selector.get_heuristic(heuristic_name)
        self.heuristic_name = heuristic_name
    
    def get_available_heuristics(self):
        """Kullanılabilir heuristic'leri döndür"""
        return self.heuristic_selector.get_all_names()
    
    def compare_heuristics_at_node(self, node, goal):
        """Bir düğümde tüm heuristic'leri karşılaştır"""
        return self.heuristic_selector.compare_heuristics(node, goal)


class AStarVariant:
    """A* algoritmasının farklı varyantları"""
    
    @staticmethod
    def weighted_astar(graph, heuristic_name='euclidean', weight=1.5):
        """
        Weighted A* - Heuristic'i ağırlıklandır
        Weight > 1: Daha hızlı ama daha az optimal
        """
        astar = AStar(graph, heuristic_name)
        
        # Orijinal heuristic fonksiyonunu sakla
        original_heuristic = astar.heuristic_func
        
        # Ağırlıklı heuristic fonksiyonu
        def weighted_heuristic(node1, node2):
            return weight * original_heuristic(node1, node2)
        
        astar.heuristic_func = weighted_heuristic
        astar.heuristic_name = f"weighted_{heuristic_name}_{weight}"
        
        return astar
    
    @staticmethod
    def bidirectional_astar(graph, heuristic_name='euclidean'):
        """
        Bidirectional A* - Her iki yönden ara
        (Basit implementasyon)
        """
        # Bu daha karmaşık bir algoritma, basitleştirilmiş versiyonu
        forward_astar = AStar(graph, heuristic_name)
        return forward_astar  # Şimdilik normal A* döndür


# Utility fonksiyonları
def path_cost(path, graph):
    """Yolun toplam maliyetini hesapla"""
    if len(path) < 2:
        return 0
    
    cost = 0
    for i in range(len(path) - 1):
        cost += graph.get_distance(path[i], path[i + 1])
    
    return cost


def smooth_path(path, graph):
    """Yolu düzleştir (basit line-of-sight kontrolü)"""
    if len(path) < 3:
        return path
    
    smoothed = [path[0]]
    
    i = 0
    while i < len(path) - 1:
        # Mevcut noktadan sonraki noktalara kadar line-of-sight var mı?
        j = len(path) - 1
        while j > i + 1:
            if has_line_of_sight(path[i], path[j], graph):
                smoothed.append(path[j])
                i = j
                break
            j -= 1
        else:
            smoothed.append(path[i + 1])
            i += 1
    
    return smoothed


def has_line_of_sight(node1, node2, graph):
    """İki nokta arasında engel var mı? (Basit Bresenham benzeri)"""
    dx = abs(node2.x - node1.x)
    dy = abs(node2.y - node1.y)
    
    x, y = node1.x, node1.y
    x_inc = 1 if node1.x < node2.x else -1
    y_inc = 1 if node1.y < node2.y else -1
    
    error = dx - dy
    
    while True:
        # Mevcut pozisyonda duvar var mı?
        current = graph.get_node(x, y)
        if current and current.node_type == 'wall':
            return False
        
        if x == node2.x and y == node2.y:
            break
        
        e2 = 2 * error
        if e2 > -dy:
            error -= dy
            x += x_inc
        if e2 < dx:
            error += dx
            y += y_inc
    
    return True